#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]
mod experimental;
mod jobs;
mod management;
mod settings;
#[cfg(test)]
mod tests;
use jobs::{Desktop, Snapshot};
use serde_json::Value;
use tauri::{Manager, State};

#[tauri::command]
fn manage(action: management::Action, state: State<'_, Desktop>) -> Result<(), String> {
    state.manage(action)
}
#[tauri::command]
fn diagnostics(state: State<'_, Desktop>) -> Result<Value, String> {
    let inner = state.inner.lock().map_err(|e| e.to_string())?;
    Ok(
        serde_json::json!({"configured_corpora":inner.settings.corpora,"configured_model":inner.settings.model,
        "last_solve_status":inner.job.status,"last_solve_timings":inner.job.result.as_ref().and_then(|r| r.get("timings")),
        "cache":{"path":inner.settings.runtime.cache_path(&state.data),"bytes":std::fs::metadata(inner.settings.runtime.cache_path(&state.data)).ok().map(|m| m.len()),"max_entries":inner.settings.runtime.cache_entries,"max_payload_bytes":inner.settings.runtime.cache_bytes},
        "note":"Identities and timings belong to the last solve, not necessarily current settings. Use Validate in Data to inspect current inputs."}),
    )
}
#[tauri::command]
fn read_report(path: String, state: State<'_, Desktop>) -> Result<Value, String> {
    read_report_file(path, &state.data)
}
fn read_report_file(path: String, data: &std::path::Path) -> Result<Value, String> {
    use std::io::Read;
    let mut bytes = Vec::new();
    std::fs::File::open(path)
        .map_err(|e| e.to_string())?
        .take(4 * 1024 * 1024 + 1)
        .read_to_end(&mut bytes)
        .map_err(|e| e.to_string())?;
    if bytes.len() > 4 * 1024 * 1024 {
        return Err("Report exceeds 4 MiB".into());
    }
    let mut value: Value = serde_json::from_slice(&bytes).map_err(|e| e.to_string())?;
    fn relocate(value: &mut Value, data: &std::path::Path) {
        if let Some(object) = value.as_object_mut() {
            for (key, value) in object {
                if [
                    "model_path",
                    "report_path",
                    "dataset_path",
                    "dictionary",
                    "unigrams",
                    "bigrams",
                    "wordnet",
                    "phrase",
                    "database",
                ]
                .contains(&key.as_str())
                {
                    if let Some(path) = value.as_str() {
                        let mut path = path.to_owned();
                        if settings::relocate_path(&mut path, data) {
                            *value = path.into();
                        }
                    }
                } else {
                    relocate(value, data);
                }
            }
        }
    }
    relocate(&mut value, data);
    Ok(value)
}

#[tauri::command]
fn bootstrap(state: State<'_, Desktop>) -> Result<Value, String> {
    state.bootstrap()
}
#[tauri::command]
fn clear_cache(state: State<'_, Desktop>) -> Result<(), String> {
    state.clear_cache()
}
#[tauri::command]
fn train_model(
    dataset: String,
    epochs: usize,
    folds: usize,
    learning_rate: Option<f64>,
    l2: Option<f64>,
    state: State<'_, Desktop>,
) -> Result<(), String> {
    state.train_model_options(
        dataset,
        anagram_core::training::Options {
            epochs,
            folds,
            learning_rate: learning_rate.unwrap_or(0.08),
            l2: l2.unwrap_or(0.002),
        },
    )
}
#[tauri::command]
fn training_status(state: State<'_, Desktop>) -> Result<experimental::Snapshot, String> {
    Ok(state
        .inner
        .lock()
        .map_err(|e| e.to_string())?
        .experiment
        .clone())
}
#[tauri::command]
fn cancel_training(state: State<'_, Desktop>) -> Result<(), String> {
    let inner = state.inner.lock().map_err(|e| e.to_string())?;
    if inner.experiment.active {
        inner.control.cancel();
    }
    Ok(())
}
#[tauri::command]
fn save_settings(settings: settings::Settings, state: State<'_, Desktop>) -> Result<(), String> {
    let mut inner = state.inner.lock().map_err(|e| e.to_string())?;
    if inner.job.active || inner.experiment.active {
        return Err("Wait for the current solve before changing settings.".into());
    }
    settings::save(&state.data, &settings)?;
    inner.settings = settings;
    inner.notice = None;
    Ok(())
}
#[tauri::command]
fn start_job(
    request: Value,
    rebuild: bool,
    extended: bool,
    generation_only: Option<bool>,
    state: State<'_, Desktop>,
) -> Result<u64, String> {
    state.start_with_mode(request, rebuild, extended, generation_only.unwrap_or(false))
}
#[tauri::command]
fn job_status(state: State<'_, Desktop>) -> Result<Snapshot, String> {
    let inner = state.inner.lock().map_err(|e| e.to_string())?;
    Ok(Snapshot {
        id: inner.job.id,
        active: inner.job.active,
        status: inner.job.status.clone(),
        result: None,
        error: inner.job.error.clone(),
    })
}
#[tauri::command]
fn job_result(id: u64, state: State<'_, Desktop>) -> Result<Option<Value>, String> {
    let inner = state.inner.lock().map_err(|e| e.to_string())?;
    if inner.job.id != id {
        return Err("That result has expired.".into());
    }
    Ok(inner.job.result.clone())
}
#[tauri::command]
fn cancel_job(id: u64, state: State<'_, Desktop>) -> Result<(), String> {
    let inner = state.inner.lock().map_err(|e| e.to_string())?;
    if inner.job.id == id && inner.job.active {
        inner.control.cancel();
    }
    Ok(())
}
#[tauri::command]
async fn choose_path(kind: String) -> Result<Option<String>, String> {
    tauri::async_runtime::spawn_blocking(move || {
        let dialog = rfd::FileDialog::new();
        let path = match kind.as_str() {
            "wordnet" | "source" => dialog.set_title("Choose source folder").pick_folder(),
            "cache" => dialog
                .add_filter("SQLite cache", &["sqlite", "db"])
                .save_file(),
            "titles" => dialog
                .add_filter("Title sources", &["txt", "gz"])
                .pick_file(),
            "dictionary" | "unigrams" | "bigrams" => {
                dialog.add_filter("Text corpus", &["txt"]).pick_file()
            }
            "phrase" => dialog
                .add_filter("Phrase database", &["db", "sqlite", "sqlite3"])
                .pick_file(),
            "model" | "training" => dialog.add_filter("Ranker JSON", &["json"]).pick_file(),
            _ => return Err("Unknown corpus type".into()),
        };
        Ok(path.map(|p| p.to_string_lossy().into_owned()))
    })
    .await
    .map_err(|e| e.to_string())?
}
fn result_text(state: &Desktop, id: u64) -> Result<String, String> {
    let inner = state.inner.lock().map_err(|e| e.to_string())?;
    if inner.job.id != id {
        return Err("That result has expired.".into());
    }
    let result = inner
        .job
        .result
        .as_ref()
        .ok_or("No completed result to copy")?;
    let mut lines = Vec::new();
    if let Some(bags) = result["bags"].as_array() {
        for bag in bags {
            lines.push(
                bag.as_array()
                    .ok_or("Invalid bag")?
                    .iter()
                    .filter_map(Value::as_str)
                    .collect::<Vec<_>>()
                    .join(" "),
            );
        }
        return Ok(lines.join("\r\n"));
    }
    let mut buckets: Vec<_> = result["buckets"]
        .as_object()
        .ok_or("Invalid result")?
        .iter()
        .map(|(count, rows)| count.parse::<usize>().map(|count| (count, rows)))
        .collect::<Result<_, _>>()
        .map_err(|_| "Invalid word count")?;
    buckets.sort_by_key(|(count, _)| *count);
    for (_, rows) in buckets {
        for row in rows.as_array().ok_or("Invalid rows")? {
            lines.push(anagram_core::format_phrase(
                &row["best_order"]
                    .as_array()
                    .ok_or("Invalid phrase")?
                    .iter()
                    .filter_map(Value::as_str)
                    .map(str::to_owned)
                    .collect::<Vec<_>>(),
            ));
        }
    }
    Ok(lines.join("\r\n"))
}
#[tauri::command]
fn copy_results(id: u64, state: State<'_, Desktop>) -> Result<(), String> {
    let text = result_text(&state, id)?;
    arboard::Clipboard::new()
        .map_err(|e| e.to_string())?
        .set_text(text)
        .map_err(|e| e.to_string())
}
#[tauri::command]
async fn export_results(
    id: u64,
    format: String,
    state: State<'_, Desktop>,
) -> Result<Option<String>, String> {
    let text = match format.as_str() {
        "txt" => result_text(&state, id)?,
        "json" => {
            let inner = state.inner.lock().map_err(|e| e.to_string())?;
            if inner.job.id != id {
                return Err("That result has expired.".into());
            }
            serde_json::to_string_pretty(inner.job.result.as_ref().ok_or("No completed result")?)
                .map_err(|e| e.to_string())?
        }
        _ => return Err("Choose text or JSON export".into()),
    };
    tauri::async_runtime::spawn_blocking(move || {
        let Some(path) = rfd::FileDialog::new()
            .set_file_name(format!("anagrams.{format}"))
            .add_filter("Results", &[&format])
            .save_file()
        else {
            return Ok(None);
        };
        std::fs::write(&path, text).map_err(|e| e.to_string())?;
        Ok(Some(path.to_string_lossy().into_owned()))
    })
    .await
    .map_err(|e| e.to_string())?
}
fn main() {
    let app = tauri::Builder::default()
        .setup(|app| {
            let executable = std::env::current_exe()?;
            let data = executable
                .parent()
                .ok_or("Executable has no parent directory")?
                .join("user-data");
            app.manage(Desktop::new(data.clone()).map_err(std::io::Error::other)?);
            tauri::WebviewWindowBuilder::from_config(app, &app.config().app.windows[0])?
                .data_directory(data)
                .build()?;
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            bootstrap,
            clear_cache,
            train_model,
            manage,
            diagnostics,
            read_report,
            training_status,
            cancel_training,
            save_settings,
            start_job,
            job_status,
            job_result,
            cancel_job,
            choose_path,
            copy_results,
            export_results
        ])
        .build(tauri::generate_context!())
        .expect("Unable to start TajsAnagrams");
    app.run(|app, event| {
        if matches!(event, tauri::RunEvent::Exit) {
            app.state::<Desktop>().shutdown();
        }
    });
}
