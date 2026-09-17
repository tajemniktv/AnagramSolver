<script lang="ts">
  import { Copy, Download } from '@lucide/svelte';
  import type { SolveResult, RankedResult, GeneratedResult, Job } from './bridge';
  let { result, job, oncopy, onexport }: { result: SolveResult | null; job: Job | null; oncopy: () => void; onexport: (format: 'txt' | 'json') => void } = $props();
  let format = $state<'txt' | 'json'>('txt');
  let generated = $derived(result && Array.isArray((result as GeneratedResult).bags) ? result as GeneratedResult : null);
  let rows = $derived(result && !generated ? Object.entries((result as RankedResult).buckets).sort(([a], [b]) => +a - +b).flatMap(([count, rows]) => rows.map((row, index) => ({ ...row, count, rank: index + 1 }))) : []);
  let page = $state(0); let total = $derived(generated ? generated.bags.length : rows.length);
  $effect(() => { void result; page = 0; });
  let counts = $derived(job?.status?.counts);
</script>
<section class="results" aria-labelledby="result-heading">
  <div class="result-heading"><h2 id="result-heading">Results</h2><div class="actions"><button disabled={!total} onclick={oncopy}><Copy size={16}/>Copy all</button><select aria-label="Export format" bind:value={format}><option value="txt">Text</option><option value="json">JSON</option></select><button disabled={!result} onclick={() => onexport(format)}><Download size={16}/>Export</button></div></div>
  <div class="result-summary" aria-live="polite">
    {#if counts}<span>Generated {counts.generated.toLocaleString()} · Analyzed {counts.deep_analyzed.toLocaleString()} · Shown {counts.shown.toLocaleString()}</span>{:else}<span>Results will appear here</span>{/if}
    {#if job?.status?.exhaustion === 'truncated'}<span>Candidate limit reached</span>{:else if job?.status?.exhaustion === 'exhausted'}<span>Search exhausted</span>{/if}
  </div>
  {#if job?.active}
    <div class="activity" role="status"><span class="pulse"></span><span>{job.status?.stage.replaceAll('_', ' ') ?? 'Starting'}…</span></div>
    <p class="hint">{counts?.orders_evaluated.toLocaleString() ?? 0} orders evaluated. Cancellation keeps confirmed counts, not partial result rows.</p>
  {/if}
  {#if total}
    <div class="actions"><button disabled={page === 0} onclick={() => page--}>Previous</button><span>Page {page + 1} / {Math.ceil(total / 100)} · {total.toLocaleString()} results</span><button disabled={(page + 1) * 100 >= total} onclick={() => page++}>Next</button></div>
    {#if generated}<p class="hint">Unranked word bags. No word-order scoring or ranked-result cache was used.</p><div class="table-scroll"><table><thead><tr><th>#</th><th>Words</th><th>Bag</th></tr></thead><tbody>{#each generated.bags.slice(page*100,(page+1)*100) as bag,i}<tr><td>{page*100+i+1}</td><td>{bag.length}</td><td>{bag.join(' ')}</td></tr>{/each}</tbody></table></div>
    {:else}<div class="table-scroll"><table><thead><tr><th>#</th><th>Words</th><th>Score</th><th>Phrase</th></tr></thead><tbody>{#each rows.slice(page*100,(page+1)*100) as row}<tr><td>{row.rank}</td><td>{row.count}</td><td class="score">{row.final.toFixed(2)}</td><td class="phrase">{row.display_phrase || row.best_order.join(' ')}</td></tr>{/each}</tbody></table></div>{/if}
    {#if !generated}<p class="hint">Scores compare candidates for this search, not probabilities. {result?.status.cache.hit ? 'Reused a verified local result.' : 'Computed on this PC.'}</p>{/if}
  {:else if !job?.active}
    <div class="empty"><h3>{job?.status?.state === 'cancelled' ? 'Solve cancelled' : result ? 'No matching phrases' : 'A new arrangement starts here'}</h3><p>{job?.status?.state === 'cancelled' ? 'The worker has stopped. Change your options or start another solve when ready.' : result ? 'Try fewer constraints, a wider word range, or a larger candidate budget.' : 'Enter your letters, add any clues, and solve locally. Your words stay on this PC.'}</p></div>
  {/if}
</section>
