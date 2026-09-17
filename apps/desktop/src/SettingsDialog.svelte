<script lang="ts">
  import { FolderOpen, X } from '@lucide/svelte';
  import { untrack } from 'svelte';
  import Experimental from './Experimental.svelte';
  import Management from './Management.svelte';
  import { bridge, message, type Settings, type CorpusKey } from './bridge';
  let { settings, directory, onsave, onclose }: { settings: Settings; directory: string; onsave: (settings: Settings) => Promise<void>; onclose: () => void } = $props();
  let draft = $state<Settings>(untrack(() => structuredClone($state.snapshot(settings))));
  let error = $state(''); let busy = $state(false); let dialog: HTMLDialogElement;
  const tabs = ['general','data','experimental','diagnostics'] as const;
  let tab = $state<typeof tabs[number]>('general');
  function navigate(event: KeyboardEvent) {
    if(busy || !['ArrowLeft','ArrowRight','Home','End'].includes(event.key)) return;
    event.preventDefault(); tab = event.key === 'Home' ? 'general' : event.key === 'End' ? 'diagnostics' : tabs[(tabs.indexOf(tab) + (event.key === 'ArrowRight' ? 1 : 3)) % tabs.length];
    dialog.querySelector<HTMLButtonElement>(`#settings-${tab}`)?.focus();
  }
  const names: [CorpusKey, string][] = [['dictionary','Dictionary'],['unigrams','Unigram counts'],['bigrams','Bigram counts'],['wordnet','WordNet dict folder'],['phrase','Phrase database (optional)']];
  $effect(() => { dialog?.showModal(); });
  async function choose(key: CorpusKey) { try { const path = await bridge.choose(key); if (path) draft.corpora[key] = path; } catch (e) { error = message(e); } }
  async function save() { busy = true; try { await onsave(draft); onclose(); } catch (e) { error = message(e); } finally { busy = false; } }
  let cacheMessage = $state('');
  const limitNames = ['max_input_bytes','max_normalized_letters','max_candidates','max_deep_analyzed','max_beam_width','max_retained_orders','timeout_ms'] as const;
  async function cachePath() { try { draft.runtime.cache_path = await bridge.choose('cache') ?? draft.runtime.cache_path; } catch(e) {error = message(e);} }
  async function clearCache() { busy = true; try { await bridge.clearCache(); cacheMessage = 'Result cache cleared. Corpora and settings were kept.'; } catch(e) { error = message(e); } finally { busy = false; } }
</script>
<dialog bind:this={dialog} onclose={onclose} oncancel={event => { if(busy) event.preventDefault(); }} class="settings-dialog">
  <div class="dialog-heading"><h2>Settings</h2><button disabled={busy} class="icon" aria-label="Close settings" onclick={onclose}><X size={20}/></button></div>
  <div role="tablist" aria-label="Settings sections" class="settings-tabs">
    {#each tabs as name}<button role="tab" id={`settings-${name}`} aria-controls={`panel-${name}`} aria-selected={tab === name} tabindex={tab === name ? 0 : -1} disabled={busy} onkeydown={navigate} onclick={() => tab = name}>{name[0].toUpperCase() + name.slice(1)}</button>{/each}
  </div>
  {#if tab === 'general'}
  <div role="tabpanel" id="panel-general" aria-labelledby="settings-general" tabindex="0">
  <p class="muted">Corpora stay on this PC. Choose existing files; nothing is uploaded.</p>
  {#each names as [key, label]}
    <label>{label}<div class="path-field"><input aria-label={label} bind:value={draft.corpora[key]}/><button class="icon" aria-label={`Browse ${label}`} onclick={() => choose(key)}><FolderOpen size={18}/></button></div></label>
  {/each}
  <label>Appearance<select bind:value={draft.theme}><option value="dark">Dark</option><option value="light">Light</option><option value="system">Follow Windows</option></select></label>
  <p class="hint">Settings and bounded result cache: <span class="path">{directory}</span></p>
  <p class="hint">Phrase tables must use unique text keys. Refinement is optional in Advanced options.</p>
  <button disabled={busy} onclick={clearCache}>Clear result cache</button>
  <p class="hint">Clear uses the currently saved cache location. Save settings before clearing a changed location.</p>
  <label>Cache path (blank: app user-data)<div class="path-field"><input bind:value={draft.runtime.cache_path}/><button onclick={cachePath}>Browse</button></div></label>
  <div class="pair"><label>Maximum cache entries<input type="number" min="1" max="4294967295" bind:value={draft.runtime.cache_entries}/></label><label>Maximum payload bytes<input type="number" min="1" max="4294967295" bind:value={draft.runtime.cache_bytes}/></label></div>
  <label class="check"><input type="checkbox" bind:checked={draft.runtime.custom_limits}/>Use custom execution limits in standard mode</label>
  {#if draft.runtime.custom_limits}<p class="hint">Blank means unlimited. Values are integer limits; timeout is in milliseconds. Extended mode bypasses these limits.</p><div class="pair">{#each limitNames as key}<label>{key.replaceAll('_',' ')}<input type="number" min={key === 'timeout_ms' ? 0 : 1} step="1" bind:value={draft.runtime.limits[key]}/></label>{/each}</div>{/if}
  {#if cacheMessage}<p role="status">{cacheMessage}</p>{/if}
  </div>
  {:else if tab === 'experimental'}
  <div role="tabpanel" id="panel-experimental" aria-labelledby="settings-experimental" tabindex="0"><Experimental bind:model={draft.model} corpora={draft.corpora} bind:busy/></div>
  {:else}
  <div role="tabpanel" id={`panel-${tab}`} aria-labelledby={`settings-${tab}`} tabindex="0">{#key tab}<Management bind:settings={draft} bind:busy diagnostics={tab === 'diagnostics'}/>{/key}</div>
  {/if}
  {#if error}<p role="alert" class="error">{error}</p>{/if}
  <div class="dialog-actions"><button disabled={busy} onclick={onclose}>Close without saving</button><button class="primary" disabled={busy} onclick={save}>Save settings</button></div>
</dialog>
<style>.settings-tabs{display:flex;gap:8px;margin-bottom:20px}.settings-tabs [aria-selected=true]{border-color:var(--accent);color:var(--accent)}[role=tabpanel]>.muted{margin-bottom:20px}</style>
