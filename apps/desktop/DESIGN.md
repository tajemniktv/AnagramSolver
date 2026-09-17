# Desktop visual specification

Concept generated with built-in Image Gen in `.codex/temp/desktop-design/concept.png`.
Direction: compact graphite utility, violet primary action, Segoe UI, fine dividers,
340px input rail and open results table; 8px controls, restrained 14px UI text.
Dark #17191e canvas, #20232b panel, #a59cff accent, #eceef5 text; light variant uses
white canvas, light-gray rail, dark text and a darker violet for contrast.

Components: App composes the shell and workflow; Advanced owns search options;
SettingsDialog owns corpus/theme editing; Results owns count/outcome/table rendering;
bridge is the only desktop transport adapter. No marketing or raster UI assets.

Allowed primary copy: TajsAnagrams, Local engine, Settings, Find the words,
Letters or phrase, Required words, Hint words, Exclude words, Advanced options,
Solve anagram, Results, Copy all, Export, Ready, Runs entirely on this PC.
Functional additions: result word-count column, export format, actual runtime
counts/status/errors, explicit limits, empty-state guidance and accessible labels.
Intentional concept deviations: native Windows titlebar instead of imitated
window buttons/hamburger; real empty/results states instead of fabricated sample
rows, scores and blank table padding. No gradient is needed for native controls.

QA must compare the concept and actual app screenshot, including layout, type,
palette, controls, spacing and responsive/keyboard behavior. The image is a design
reference only, never an executable result fixture or a screenshot of the app.

Dated visual QA evidence is recorded in `VERIFICATION.md`.
