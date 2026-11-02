# VideoSplat Versioning Policy

VideoSplat uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html) and Git tags in the form `vMAJOR.MINOR.PATCH`.

## Branch and release policy

- `main` is the current official development line and should remain in a releasable state.
- Every formal release must have an annotated Git tag and a matching GitHub Release.
- The tag must point to the reviewed release commit containing the code, documentation, and changelog for that version.
- `README.md`, `README_EN.md`, `CHANGELOG.md`, and the relevant version document must be updated together before tagging.
- Release dates must be recorded only when the tag and release are actually created; unreleased versions use `Unreleased`.

## Version increments

- **MAJOR**: incompatible workflow, data-format, API, or command-line changes.
- **MINOR**: backward-compatible functionality.
- **PATCH**: backward-compatible fixes or documentation corrections.
- Major incompatible changes must include a migration guide describing affected inputs, outputs, configuration, and upgrade steps.

## Historical versions

- Historical source is retrieved through Git tags; complete copies of old source trees are not stored in version-named folders on `main`.
- Version documentation must describe the architecture and feature scope at the time of release.
- Installation instructions for a historical version must refer to dependencies that actually match that tagged source.
- A published tag must not be moved or overwritten. Corrections are released under a new version number.

## Release checklist

1. Confirm the intended release commit and review all uncommitted changes.
2. Run available tests and validate README, demo, and documentation links.
3. Update both READMEs, the changelog, version documentation, and release notes.
4. Create an annotated `vMAJOR.MINOR.PATCH` tag on the reviewed release commit.
5. Push the commit and tag, then create the matching GitHub Release from the prepared notes.
