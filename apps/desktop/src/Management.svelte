<script lang="ts">
  import { onMount } from 'svelte';
  import { bridge, message, type Settings } from './bridge';
  let { settings = $bindable(), busy = $bindable(false), diagnostics = false }: {settings: Settings; busy?: boolean; diagnostics?: boolean} = $props();
  let source = $state(''); let wiktionary = $state(false); let wikipedia = $state(false);
  let base = $state(''); let unigrams = $state(''); let titles = $state('');
  async function pick(kind: 'dictionary' | 'unigrams' | 'titles') { try { const path = await bridge.choose(kind); if(!path) return; if(kind === 'dictionary') base = path; else if(kind === 'unigrams') unigrams = path; else titles += (titles ? '\n' : '') + path; } catch(e) { error = message(e); } }
  let report = $state<Record<string, unknown> | null>(null); let error = $state(''); let cancelling = $state(false);
  let timer: ReturnType<typeof setTimeout>; let disposed = false; let revision = 0;
  async function refresh(restore = false) {
    if(disposed) return;
    const current = ++revision;
    try { const status = await bridge.maintenanceStatus(); if(disposed || current !== revision) return; busy = status.active; error = restore ? '' : status.error ?? ''; report = status.report;
      if(busy) timer = setTimeout(() => refresh(),500); else cancelling = false;
    } catch(e) { if(disposed || current !== revision) return; error = message(e); if(busy) timer = setTimeout(() => refresh(),1000); }
  }
  async function run(action: unknown) { if(busy) return; ++revision; clearTimeout(timer); busy = true; error = ''; report = null; try { await bridge.manage(action); await refresh(); } catch(e) { if(disposed) return; error = message(e); busy = false; } }
  async function inspect() { try { report = await bridge.diagnostics(); } catch(e) { error = message(e); } }
  async function browse() { try { source = await bridge.choose('source') ?? source; } catch(e) { error = message(e); } }
  async function reopen() { try { const path = await bridge.choose('training'); if(!path) return; const saved = await bridge.readReport(path) as Record<string, unknown>;
    const c = saved?.corpora as Record<string, unknown> | undefined;
    if(!c || !['dictionary','unigrams','bigrams','wordnet','phrase'].every(key => typeof c[key] === 'string')) throw new Error('Choose a prepared corpus-set manifest.json');
    report = saved; error = '';
  } catch(e) { error = message(e); } }
  async function cancel() { try { await bridge.cancelTraining(); cancelling = true; } catch(e) { error = message(e); } }
  function select() { if(report?.corpora) settings.corpora = structuredClone(report.corpora as Settings['corpora']); }
  onMount(() => { if(diagnostics) inspect(); else refresh(true); return () => {disposed = true; clearTimeout(timer);}; });
</script>
{#if diagnostics}
  <h3>Diagnostics</h3><p class="hint">Configured paths, last solve identities (SHA-256), timings in milliseconds, effective budgets and cache flags. Last-solve evidence may predate settings changes. Nothing is uploaded.</p>
  <button onclick={inspect}>Refresh diagnostics</button>
{:else}
  <h3>Data management</h3>
  <p class="hint">Validate the paths in General, or prepare a new corpus set. Updates never overwrite your current set. Selecting a prepared set changes the settings draft; Save settings activates it.</p>
  <button disabled={busy} onclick={() => run({kind:'validate',corpora:$state.snapshot(settings.corpora),model:settings.model})}>Validate selected corpora and model</button>
  <h3>Download / prepare / update</h3>
  <p class="hint">Leave the source folder blank to download the dictionary, Norvig counts and WordNet from their original providers. Optional Wikimedia title dumps can be large. Each source is limited to 2 GiB; the whole operation to one hour. Downloads are cancellable and fail after 30 seconds without network data. No downloads start until you press the button.</p>
  <label>Offline source folder (optional)<div class="path-field"><input disabled={busy} bind:value={source}/><button disabled={busy} onclick={browse}>Browse</button></div></label>
  <p class="hint">Offline filenames: base.txt, count_1w.txt, count_2w.txt, wordnet.tar.gz; optional enwiktionary.gz and enwiki.gz.</p>
  <label class="check"><input disabled={busy} type="checkbox" bind:checked={wiktionary}/>Include Wiktionary phrases</label>
  <label class="check"><input disabled={busy} type="checkbox" bind:checked={wikipedia}/>Include Wikipedia phrases</label>
  <p class="hint">Phrase preparation automatically uses up to 8 parsing workers, bounded batches and a 64 MiB database cache. SQLite writes remain serial; cancellation stays available. The completed report includes worker and database-update counts.</p>
  <button disabled={busy} onclick={() => run({kind:'prepare',source,wiktionary,wikipedia})}>{source ? 'Prepare new set from local sources' : 'Download and prepare new set'}</button>
  <button disabled={busy} onclick={reopen}>Open existing set manifest</button>
  <h3>Independent preparation tools</h3>
  <p class="hint">Prepare only a dictionary or phrase database from arbitrary local files. Outputs get new folders in user-data/prepared; select the result and Save settings to activate it.</p>
  <label>Base dictionary<div class="path-field"><input disabled={busy} bind:value={base}/><button disabled={busy} onclick={() => pick('dictionary')}>Browse</button></div></label>
  <label>Unigram counts<div class="path-field"><input disabled={busy} bind:value={unigrams}/><button disabled={busy} onclick={() => pick('unigrams')}>Browse</button></div></label>
  <button disabled={busy || !base || !unigrams} onclick={() => run({kind:'dictionary',base,unigrams})}>Prepare dictionary only</button>
  <label>Title files (text/gzip, one path per line)<textarea disabled={busy} rows="3" bind:value={titles}></textarea></label>
  <button disabled={busy} onclick={() => pick('titles')}>Add title file</button>
  <button disabled={busy || !titles.trim()} onclick={() => run({kind:'phrases',sources:titles.split('\n').map(s => s.trim()).filter(Boolean)})}>Build phrase database only</button>
  {#if typeof report?.dictionary === 'string'}<button onclick={() => settings.corpora.dictionary = String(report?.dictionary)}>Select prepared dictionary</button>{/if}
  {#if typeof report?.database === 'string'}<button onclick={() => settings.corpora.phrase = String(report?.database)}>Select phrase database</button>{/if}
  {#if report?.corpora}<p class="hint">Set report available. Previous files remain untouched. After selecting an older set, use Validate to recheck its files.</p><button disabled={busy} onclick={select}>Select prepared set</button>{/if}
{/if}
{#if busy}<p role="status">{cancelling ? 'Cancelling; waiting for the worker…' : 'Working locally. Settings are locked until completion.'}</p><button disabled={cancelling} onclick={cancel}>Cancel</button>{/if}
{#if error}<p class="error" role="alert">{error}</p>{/if}
{#if report}<h3>Report</h3><pre>{JSON.stringify(report,null,2)}</pre>{/if}
<style>h3{margin:18px 0 12px}pre{white-space:pre-wrap;overflow-wrap:anywhere;max-height:360px;overflow:auto;font-size:12px;padding:12px;background:var(--bg)}</style>
