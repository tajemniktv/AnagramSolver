<script lang="ts">
  import { onMount } from 'svelte';
  import { FolderOpen } from '@lucide/svelte';
  import { bridge, message, type TrainingStatus, type Settings } from './bridge';
  let { model = $bindable(''), busy = $bindable(false), corpora }: { model?: string; busy?: boolean; corpora: Settings['corpora'] } = $props();
  let cases = $state(''); let buildReport = $state<Record<string, unknown> | null>(null);
  let dataset = $state(''); let epochs = $state(80); let folds = $state(5);
  let learningRate = $state(0.08); let l2 = $state(0.002); let retainedOrders = $state(56); let phraseBonus = $state(10);
  let items = $state(''); let rankingReport = $state<Record<string, unknown> | null>(null);
  let status = $state<TrainingStatus | null>(null); let error = $state(''); let cancelling = $state(false);
  let timer: ReturnType<typeof setTimeout>; let disposed = false;
  async function browse(kind: 'model' | 'training') { try { const path = await bridge.choose(kind); if(path) { if(kind === 'model') model = path; else dataset = path; } } catch(e) { error = message(e); } }
  async function refresh() {
    if(disposed) return;
    try { const next = await bridge.maintenanceStatus(); if(disposed) return; busy = next.active;
      status = {active:next.active,error:next.error,report:next.report?.model_path ? next.report as unknown as TrainingStatus['report'] : null};
      if(next.report?.dataset_path) { dataset = String(next.report.dataset_path); buildReport = next.report; }
      if(next.report?.indices) rankingReport = next.report;
      if(next.active) timer = setTimeout(refresh, 300); else cancelling = false; }
    catch(e) { error = message(e); if(busy) timer = setTimeout(refresh, 1000); }
  }
  async function train(event: SubmitEvent) {
    event.preventDefault(); if(busy) return; busy = true; error = ''; cancelling = false;
    try { await bridge.train(dataset, epochs, folds, learningRate, l2); await refresh(); } catch(e) { error = message(e); busy = false; }
  }
  async function cancel() { try { await bridge.cancelTraining(); cancelling = true; } catch(e) { error = message(e); } }
  async function build() { busy = true; error = ''; buildReport = null; try { await bridge.manage({kind:'build_training',cases,corpora:$state.snapshot(corpora),options:{retained_orders:retainedOrders,phrase_bonus_max:phraseBonus}}); await refresh(); } catch(e) { error = message(e); busy = false; } }
  async function rank() { busy = true; error = ''; rankingReport = null; try { await bridge.manage({kind:'rank_model',model,items}); await refresh(); } catch(e) { error = message(e); busy = false; } }
  async function browseItems() { try { items = await bridge.choose('training') ?? items; } catch(e) { error = message(e); } }
  async function browseCases() { try { cases = await bridge.choose('training') ?? cases; } catch(e) { error = message(e); } }
  async function openReport() {
    try { const path = await bridge.choose('training'); if(!path) return;
      const report = await bridge.readReport(path) as NonNullable<TrainingStatus['report']>;
      if(!report || typeof report.model_path !== 'string' || !report.held_out || !report.baseline || ![report.held_out.recall1,report.held_out.mrr,report.baseline.recall1,report.baseline.mrr].every(n => typeof n === 'number' && Number.isFinite(n))) throw new Error('Choose a saved training .report.json file');
      status = {active:false,error:null,report};
    } catch(e) { error = message(e); }
  }
  onMount(() => { refresh(); return () => { disposed = true; clearTimeout(timer); }; });
</script>
<h3>Experimental order ranking</h3>
<p class="hint">Optional research feature—not required for solving. There is no proven quality improvement yet. Nothing trains, downloads or uploads automatically.</p>
<label>Selected model (optional)<div class="path-field"><input disabled={busy} bind:value={model} placeholder="Leave blank for standard ranking"/><button disabled={busy} class="icon" aria-label="Browse model" onclick={() => browse('model')}><FolderOpen size={18}/></button></div></label>
<p class="hint">Save settings to apply this selection. A model chooses among retained word orders; it does not generate words or relax letter constraints.</p>
<h3>Build a training set</h3>
<p class="hint">Choose a JSON array of labelled cases, for example <code>{'[{"answer":"we are home","acceptable_orders":["we are home"]}]'}</code>. Use many distinct word bags. This uses the selected WordNet and optional phrase corpus to enumerate 2–6-word orders. Cases without retained positive and negative examples are reported as skipped, not silently labelled. Input limit: 16 MiB; build deadline: one hour.</p>
<label>Labelled cases<div class="path-field"><input disabled={busy} bind:value={cases}/><button disabled={busy} onclick={browseCases}>Browse</button></div></label>
<button disabled={busy || !cases} onclick={build}>Build training dataset</button>
<div class="pair"><label>Retained orders<input disabled={busy} type="number" min="2" step="1" bind:value={retainedOrders}/></label><label>Training phrase bonus<input disabled={busy} type="number" min="0" step="0.1" bind:value={phraseBonus}/></label></div>
{#if buildReport}<p class="hint">Built {String(buildReport.groups)} groups. Dataset selected below.</p><details><summary>Skipped cases and output</summary><pre>{JSON.stringify(buildReport,null,2)}</pre></details>{/if}
<h3>Train from prepared examples</h3>
<p class="hint">Use the dataset built above or import native <code>ranker-build</code> output. It must contain labelled positive and negative orders grouped by word bag—not a dictionary or raw text corpus.</p>
<p class="hint">Cost limits: one CPU worker, 60 seconds, dataset up to 4 MiB and 1,000 groups. The resulting model has only 18 weights and is a small JSON file. Creating and labelling good examples is the bigger investment.</p>
<form onsubmit={train}>
  <fieldset disabled={busy}>
    <label>Prepared training dataset<div class="path-field"><input required bind:value={dataset} placeholder="Choose a local groups JSON file"/><button type="button" class="icon" aria-label="Browse training dataset" onclick={() => browse('training')}><FolderOpen size={18}/></button></div></label>
    <div class="pair"><label>Epochs<input type="number" required min="1" max="200" step="1" bind:value={epochs}/></label><label>Validation folds<input type="number" required min="2" max="10" step="1" bind:value={folds}/></label></div>
    <button type="submit" disabled={!dataset}>Train and evaluate locally</button>
    <div class="pair"><label>Learning rate<input required type="number" min="0.000001" step="any" bind:value={learningRate}/></label><label>L2 regularization<input required type="number" min="0" step="any" bind:value={l2}/></label></div>
  </fieldset>
</form>
{#if busy}<p class="hint" role="status">{cancelling ? 'Cancellation requested; waiting for the worker to stop…' : 'Building or evaluating examples locally…'}</p><button disabled={cancelling} onclick={cancel}>Cancel</button>{/if}
<p class="hint">Quality reports are saved beside trained models as .report.json files.</p><button disabled={busy} onclick={openReport}>Open saved quality report</button>
<h3>Rank supplied feature records</h3>
<p class="hint">Run the selected model on a JSON array of key, features (18 numbers), and baseline_score records. This does not train or activate the model.</p>
<label>Feature records<div class="path-field"><input disabled={busy} bind:value={items}/><button disabled={busy} onclick={browseItems}>Browse</button></div></label>
<button disabled={busy || !items || !model} onclick={rank}>Rank records</button>
{#if rankingReport}<pre>{JSON.stringify(rankingReport,null,2)}</pre>{/if}
{#if error || status?.error}<p class="error" role="alert">{error || status?.error}</p>{/if}
{#if status?.report && !busy}
  <h3>Held-out results</h3>
  <p class="hint">{status.report.held_out.groups} groups evaluated without training on their word bags. Higher values are better; compare against the baseline before choosing a model.</p>
  <table><thead><tr><th>Ranker</th><th>Correct first</th><th>Mean reciprocal rank</th></tr></thead><tbody>
    <tr><td>Baseline</td><td>{(100 * status.report.baseline.recall1).toFixed(1)}%</td><td>{status.report.baseline.mrr.toFixed(3)}</td></tr>
    <tr><td>Trained (held-out)</td><td>{(100 * status.report.held_out.recall1).toFixed(1)}%</td><td>{status.report.held_out.mrr.toFixed(3)}</td></tr>
  </tbody></table>
  <p class="hint path">Saved: {status.report.model_path}</p>
  <p class="hint">The saved model is fitted on all groups; its training-set performance is not presented as a quality estimate. It has not been enabled automatically.</p>
  <button onclick={() => { if(status?.report) model = status.report.model_path; }}>Select this model</button>
{/if}
<style>h3{margin:16px 0 10px}form{margin-bottom:14px}table{margin-bottom:14px;font-size:12px}pre{white-space:pre-wrap;overflow-wrap:anywhere;max-height:240px;overflow:auto}</style>
