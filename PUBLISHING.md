# ComfyUI Manager and Registry publishing

GitHub hosting and an official Registry listing are separate steps. Git installation
works after this repository is public; the listing depends on registration/review.

1. Create or select your publisher at https://registry.comfy.org/.
2. Set `PublisherId` in `pyproject.toml` to the actual Registry publisher ID.
   It is intentionally empty until verified; a GitHub username is not proof of a Registry identity.
3. Generate a publishing key for that publisher. In this repository's Settings >
   Secrets and variables > Actions, save it as `REGISTRY_ACCESS_TOKEN`.
4. Run the **Publish to Comfy Registry** GitHub Actions workflow manually.
5. Verify the package/version on the Registry and refresh ComfyUI Manager's catalog.

Legacy Manager catalogs also accept pull requests adding the repository to
ComfyUI-Manager's `custom-node-list.json`. A submitted PR is not a merged listing.

Official guide: https://docs.comfy.org/registry/publishing
