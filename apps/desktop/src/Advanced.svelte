<script lang="ts">
  import type { SolveRequest } from './bridge';
  let { request = $bindable(), disabled = false, custom = false, generationOnly = false, extended = $bindable(false), rebuild = $bindable(false) }: { request: SolveRequest; disabled?: boolean; custom?: boolean; generationOnly?: boolean; extended?: boolean; rebuild?: boolean } = $props();
  let relaxed = $derived(extended || custom);
  function preset(mode: string) {
    if (mode === 'custom') return;
    extended = mode === 'exhaustive';
    request.generation.candidate_budget = mode === 'quick' ? 10000 : mode === 'exhaustive' ? 0 : 100000;
    request.generation.min_words = 2; request.generation.min_word_length = 2; request.generation.min_zipf = 2.7;
  }
</script>
<details class="advanced">
  <summary>Advanced options</summary>
  <fieldset {disabled}>
    <label>Search preset<select value="custom" onchange={e => { preset(e.currentTarget.value); e.currentTarget.value = 'custom'; }}><option value="custom">Custom options</option><option value="quick">Quick (10,000 candidates)</option><option value="balanced">Balanced (100,000 candidates)</option><option value="exhaustive">Exhaustive generation (unlimited)</option></select></label>
    <label class="check"><input type="checkbox" bind:checked={extended}/>Extended local search (no time or candidate cap)</label>
    {#if extended}<p class="hint">May use substantial memory and CPU until cancelled. Candidate budget 0 enumerates all bags, not all deeply ranked results. Ordering still supports at most 10 words.</p>{/if}
    <label class="check"><input type="checkbox" bind:checked={rebuild}/>Force recompute (ignore saved result)</label>
    <label>Ranking workers (0 = automatic)<input type="number" min="0" max="32" bind:value={request.workers}/></label>
    <label class="check"><input type="checkbox" bind:checked={request.refine}/>Refine word orders (bounded local search)</label>
    <div class="pair">
      <label>Minimum words<input type="number" min="1" max={generationOnly ? 9007199254740991 : 10} bind:value={request.generation.min_words}/></label>
      <label>Maximum words<input type="number" min="1" max={generationOnly ? 9007199254740991 : 10} bind:value={request.generation.max_words}/></label>
      <label>Minimum length<input type="number" min="1" max={relaxed ? 9007199254740991 : 40} bind:value={request.generation.min_word_length}/></label>
      <label>Maximum length<input type="number" min="1" max={relaxed ? 9007199254740991 : 40} bind:value={request.generation.max_word_length}/></label>
    </div>
    <label>Candidate budget<input type="number" min={relaxed ? 0 : 1} max={relaxed ? 9007199254740991 : 250000} step="1" required bind:value={request.generation.candidate_budget}/></label>
    <div class="pair">
      <label>Deep per group<input type="number" min="1" max={relaxed ? 9007199254740991 : 10000} bind:value={request.deep_per_group}/></label>
      <label>Results per group<input type="number" min="1" max={relaxed ? 9007199254740991 : 100} bind:value={request.result_limit_per_group}/></label>
    </div>
    <label>Search strategy<select bind:value={request.generation.strategy}><option value="prefix">Prefix (default)</option><option value="diverse">Diverse</option></select></label>
    <label>Hint mode<select bind:value={request.generation.hint_mode}><option value="any">At least one hint</option><option value="exactly_one">Exactly one distinct hint</option></select></label>
    <label>Short words<select bind:value={request.generation.short_policy}><option value="common">Common words</option><option value="all">All short words</option><option value="none">Only explicit whitelist</option></select></label>
    <label>Minimum frequency (Zipf)<input type="number" min="0" max={relaxed ? undefined : 10} step="0.1" bind:value={request.generation.min_zipf}/></label>
    <label>Forbidden letters<input bind:value={request.generation.forbid_chars}/></label>
    <label>Extra short words<input value={(request.generation.extra_short_words ?? []).join(', ')} oninput={e => request.generation.extra_short_words = e.currentTarget.value.split(/[\s,]+/).filter(Boolean)}/></label>
    <label>Exclude patterns (one per line)<textarea rows="2" value={(request.generation.exclude_regex ?? []).join('\n')} oninput={e => request.generation.exclude_regex = e.currentTarget.value.split('\n').filter(Boolean)}></textarea></label>
    <p class="hint">Case-insensitive Rust regex. Look-around and backreferences are not supported.</p>
    <label class="check"><input type="checkbox" bind:checked={request.generation.allow_repeat}/>Allow repeated words</label>
    <label class="check"><input type="checkbox" bind:checked={request.positive_bigrams}/>Use positive bigram evidence</label>
    <label class="check"><input type="checkbox" bind:checked={request.deep_all}/>Deep-analyze every candidate{relaxed ? '' : ' (limit 10,000)'}</label>
    <label>Ordering<select bind:value={request.order_mode}><option value="auto">Automatic</option><option value="exact">Exact</option><option value="beam">Beam</option></select></label>
    <div class="pair">
      <label>Beam width<input type="number" min="1" max={relaxed ? 9007199254740991 : 512} bind:value={request.beam_width}/></label>
      <label>Exact up to words<input type="number" min="1" max="10" bind:value={request.exact_max_words}/></label>
      <label>Retained orders<input type="number" min="1" max={relaxed ? 9007199254740991 : 128} bind:value={request.retained_orders}/></label>
      <label>Phrase shortlist<input type="number" min="1" max={relaxed ? 9007199254740991 : 10000} bind:value={request.phrase_rescore_top}/></label>
    </div>
    <label>Maximum phrase bonus<input type="number" min="0" max={relaxed ? 9007199254740991 : 100} step="0.5" bind:value={request.phrase_bonus_max}/></label>
    <p class="hint">{extended ? 'Extended search enabled; cancellation stays available.' : custom ? 'Custom execution limits from Settings apply; work beyond them is rejected, not silently clamped.' : 'Standard limits: 40 letters, 250,000 candidates, 10,000 deep analyses, 120 seconds. Work is rejected, never silently clamped.'}</p>
  </fieldset>
</details>
