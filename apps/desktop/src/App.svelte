<script lang="ts">
  import { onMount } from 'svelte';
  import { Settings as SettingsIcon, Sun, Moon, Square } from '@lucide/svelte';
  import { bridge, message, type Bootstrap, type Job, type Settings, type SolveResult } from './bridge';
  import Advanced from './Advanced.svelte';
  import SettingsDialog from './SettingsDialog.svelte';
  import Results from './Results.svelte';
  let boot = $state<Bootstrap | null>(null); let settings = $state<Settings | null>(null);
  let job = $state<Job | null>(null); let result = $state<SolveResult | null>(null);
  let error = $state(''); let toast = $state(''); let showSettings = $state(false); let busy = $state(false); let cancelling = $state(false);
  let poll: ReturnType<typeof setTimeout>; let disposed = false;
  let required = $state(''); let hints = $state(''); let excluded = $state('');
  let extended = $state(false); let rebuild = $state(false);
  let generationOnly = $state(false);
  let active = $derived(busy || !!job?.active);
  let letters = $derived(settings?.request.generation.text.normalize('NFKD').replace(/[^a-z]/gi, '').length ?? 0);
  const words = (text: string) => text.split(/[\s,]+/).filter(Boolean);
  function applyTheme(theme: Settings['theme']) { document.documentElement.dataset.theme = theme; }
  async function load() {
    try { boot = await bridge.bootstrap(); settings = boot.settings; settings.request.workers ??= 1; settings.request.refine ??= false; applyTheme(settings.theme);
      required = settings.request.generation.required?.join('\n') ?? ''; hints = settings.request.generation.hints?.join('\n') ?? ''; excluded = settings.request.generation.excluded?.join('\n') ?? '';
      error = boot.notice ?? boot.corpus_error ?? ''; await refresh();
    } catch (e) { error = `Desktop connection unavailable: ${message(e)}. Launch the installed TajsAnagrams app, not a browser preview.`; }
  }
  async function refresh() {
    if (disposed) return;
    try { job = await bridge.status(); if (!job.active && job.id) { cancelling = false; result = await bridge.result(job.id); if (job.error && job.status?.state !== 'cancelled') error = job.error; }
      if (job.active) poll = setTimeout(refresh, 250);
    } catch (e) { error = message(e); }
  }
  async function start(event: SubmitEvent) {
    event.preventDefault(); if (!settings || active) return;
    settings.request.generation.required = words(required); settings.request.generation.hints = words(hints); settings.request.generation.excluded = words(excluded);
    busy = true; error = ''; toast = ''; cancelling = false; result = null;
    try { await bridge.start($state.snapshot(settings.request), rebuild, extended, generationOnly); await refresh(); } catch (e) { error = message(e); } finally { busy = false; }
  }
  async function cancel() { try { if (job?.active) { await bridge.cancel(job.id); cancelling = true; } } catch (e) { error = message(e); } }
  async function save(next: Settings) { const saved = $state.snapshot(next); await bridge.save(saved); settings = saved; applyTheme(saved.theme); boot = await bridge.bootstrap(); error = boot.corpus_error ?? ''; toast = 'Settings saved'; }
  async function toggleTheme() { if (!settings) return; const next = structuredClone($state.snapshot(settings)); next.theme = next.theme === 'light' ? 'dark' : 'light'; try { await save(next); } catch (e) { error = message(e); } }
  async function copy() { try { if (job) { await bridge.copy(job.id); toast = 'Phrases copied to clipboard'; } } catch (e) { error = message(e); } }
  async function exportRows(format: 'txt' | 'json') { try { if (job) { const path = await bridge.export(job.id, format); if (path) toast = `Exported to ${path}`; } } catch (e) { error = message(e); } }
  onMount(() => { load(); return () => { disposed = true; clearTimeout(poll); }; });
</script>
<div class="app-shell">
  <header><h1 style="display:flex;align-items:center;gap:10px"><img src="/logo.png" alt="" width="34" height="34"/>TajsAnagrams</h1><div class="header-actions"><span class="local"><i></i>Local engine</span><button class="icon" disabled={active || !settings} aria-label="Toggle appearance" onclick={toggleTheme}>{#if settings?.theme === 'light'}<Moon size={19}/>{:else}<Sun size={19}/>{/if}</button><button disabled={active || !settings} onclick={() => showSettings = true}><SettingsIcon size={17}/>Settings</button></div></header>
  <main>
    <aside>
      <h2>Find the words</h2>
      {#if settings}
      <form onsubmit={start}>
        <fieldset disabled={active}>
          <label>Letters or phrase<textarea class="target" required maxlength={extended || settings.runtime.custom_limits ? undefined : 4096} bind:value={settings.request.generation.text} placeholder="I AM TESTING ANAGRAMS" spellcheck="false"></textarea></label>
          <p class="hint counter">{letters} letters{extended ? ' · extended local search' : settings.runtime.custom_limits ? ' · custom execution limits' : ' · maximum 40'}</p>
          <label>Required words <span>(one per line)</span><textarea rows="2" bind:value={required} placeholder="e.g. testing"></textarea></label>
          <label>Hint words <span>(one per line)</span><textarea rows="2" bind:value={hints} placeholder="At least one should appear"></textarea></label>
          <label>Exclude words <span>(one per line)</span><textarea rows="2" bind:value={excluded} placeholder="Words to leave out"></textarea></label>
        </fieldset>
        <Advanced bind:request={settings.request} bind:extended bind:rebuild {generationOnly} custom={settings.runtime.custom_limits} disabled={active}/>
        <label class="check"><input disabled={active} type="checkbox" bind:checked={generationOnly}/>Generate word bags only (skip ranking)</label>
        {#if active}<button class="primary solve" type="button" disabled={busy || !job?.active || cancelling} onclick={cancel}><Square size={16}/>Cancel solve</button>{:else}<button class="primary solve" type="submit" disabled={!letters || (!extended && !settings.runtime.custom_limits && letters > 40)}>{generationOnly ? 'Generate word bags' : 'Solve anagram'}</button>{/if}
      </form>
      {:else}<p class="muted">Connecting to the local engine…</p>{/if}
    </aside>
    <div class="workspace">
      {#if error}<div class="error" role="alert">{error}</div>{/if}
      <Results {result} {job} oncopy={copy} onexport={exportRows}/>
    </div>
  </main>
  <footer role="status"><span>{active && cancelling ? 'Cancellation requested. Waiting for the worker to stop…' : toast || (active ? 'Solving locally' : job?.status?.state.replaceAll('_', ' ') ?? (job?.error ? 'Solve failed' : 'Ready'))}</span><span>Runs entirely on this PC · {extended ? 'extended search' : settings?.runtime.custom_limits ? 'custom execution limits' : '120 s job limit'}</span></footer>
</div>
{#if showSettings && settings && boot}<SettingsDialog {settings} directory={boot.data_directory} onsave={save} onclose={() => showSettings = false}/>{/if}
