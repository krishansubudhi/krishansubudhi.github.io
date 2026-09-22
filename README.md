# krishansubudhi.github.io

Source for [Krishan's Tech Blog](https://krishansubudhi.github.io), a Jekyll site
published with GitHub Pages.

## How this site is built

GitHub Pages can publish a repository in one of two ways, and which one is active
is a repository setting: **Settings → Pages → Build and deployment → Source**.

- **Deploy from a branch** (the legacy builder). GitHub runs Jekyll server-side
  with its own `github-pages` gem set. It ignores this repository's `Gemfile`,
  only supports Jekyll 3, and supports no third-party plugins. It also honours a
  root `.nojekyll` file by skipping Jekyll entirely and publishing the repository
  verbatim.
- **GitHub Actions**. The workflow in `.github/workflows/` builds the site with
  the exact gems pinned in `Gemfile.lock` and uploads the result as a Pages
  artifact. Any Jekyll version and any plugin works.

These two are mutually exclusive, and mixing them up breaks the site silently:
committing a tree that only the Actions builder can handle while the source is
still set to "Deploy from a branch" means the legacy builder publishes the raw
repository, so every page serves its unrendered front matter. If the site ever
starts showing `---` and YAML at the top of the page, that is what happened.

## Local preview

    bundle install
    bundle exec jekyll serve

## Conventions

- Posts live in `_posts/` as `YYYY-MM-DD-slug.md`.
- Permalinks are part of the site's public contract. Published URLs must not
  change, including across a theme switch.
