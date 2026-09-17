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

## Completed comparison, 2026-09-17

Compared the generated concept with the user's maximized screenshots and native
captures of the installed 1200px window. The graphite rail/canvas, violet action,
Segoe UI typography, restrained borders, spacing and two-pane hierarchy match the
direction. The implemented denser controls and extra word-count column are useful
intentional deviations; result counts and scores are actual engine output.
Light mode retains readable contrast; Follow Windows restores the dark palette on
this PC. Keyboard zoom through 150% increases text and activates the compact
stacked results toolbar without clipping controls horizontally. Independent pane
scrolling keeps long advanced options and results accessible. Focus rings and
keyboard select/submission were exercised. Reduced-motion behavior is provided
by the CSS media rule; Windows accessibility settings were not changed for QA.
