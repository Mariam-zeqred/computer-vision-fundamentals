# TODO

## Open (QC 2026-10-13)

- [ ] `README.md` is 58 lines; STYLE.md asks for 80–200 with an explicit "Next steps" section. Add "Next steps" (lesson 2 scope, notebook policy) and expand "What the outputs look like" with the printed numbers, without padding.
- [ ] Lesson 1 has one Mermaid diagram (section 3, colour-space map). The 2026-10-12 LOG.md entry says two. Either add a second (the worked-example pipeline: image → HSV → hue histogram → valley → mask → patch check) or correct the LOG entry.
- [ ] The only commit before this QC is authored `Claude <noreply@anthropic.com>` (the sandbox's git identity). Decide whether future commits should carry Mariam's name/email (set `git config user.*` in the sandbox); do not rewrite history.
- [ ] Mariam: confirm the attribution in `data/README.md` (Rachel Michetti, Pikolo Espresso Bar, CC0) against the scikit-image `skimage.data.coffee` docstring. The sandbox cannot reach scikit-image.org or the scikit-image GitHub repo, so it was not checked externally.
- [ ] Lesson 2 (filtering, edges, morphology) should open by cleaning the lesson 1 saucer mask, as WALKTHROUGH.md promises.
