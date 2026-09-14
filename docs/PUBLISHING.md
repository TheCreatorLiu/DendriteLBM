# Publishing to GitHub

This directory is a self-contained repository payload. It includes source, documentation, the MIT code license, citation metadata and ten original figure PDFs with derived PNG previews. Runtime caches, dependency copies, simulation outputs and author machine paths are excluded.

Create an empty GitHub repository under the intended owner, then run from this directory:

```bash
git init -b main
git add .
git commit -m "Prepare three Warp dendrite models and manuscript figures"
```

Set the remote to the exact URL shown by GitHub for that repository, then push `main`:

```bash
# Replace the example values with the actual owner and repository name.
git remote add origin https://github.com/OWNER/REPOSITORY.git
git push -u origin main
```

The packaged files have no individual GitHub remote configured. Update repository URLs in documentation and CITATION.cff once the destination exists. Add publication DOI metadata when it becomes available. The CPU workflow will run after a push; its existence does not imply it has already passed on GitHub.

The code uses MIT, as selected by the author. Third-party comparison data visible in the supplied figure PDFs retain their original attribution. The local validation and manuscript-alignment notes describe the release's verified scope.
