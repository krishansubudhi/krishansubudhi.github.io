# frozen_string_literal: true

source "https://rubygems.org"

# This site is NO LONGER published by the default GitHub Pages builder. The Chirpy
# theme dropped `remote_theme` support and requires Jekyll 4, which that builder does
# not run, so the site is built by .github/workflows/pages-deploy.yml instead.
# Repository Settings -> Pages -> Source must be set to "GitHub Actions"; if it is ever
# set back to "Deploy from a branch" the site silently stops updating.
gem "jekyll-theme-chirpy", "~> 7.6"

# Why no `gem "github-pages"`? It was tried, and it is what produced the 10 Dependabot
# advisories cleared earlier: the meta-gem hard-depends on jekyll-remote-theme (which
# caps rubyzip < 3.0) and on html-pipeline, and resolving through it reintroduces more
# advisories than it clears. Chirpy pulls Jekyll 4.3 plus five small first-party
# plugins (paginate, seo-tag, archives, sitemap, include-cache) and none of that set.
# Keep it that way. Run `bundle update` to pick up patch releases.

# Used only by the "Test site" step of the deploy workflow.
gem "html-proofer", "~> 5.0", group: :test

# Windows does not include zoneinfo files, so bundle the tzinfo-data gem
platforms :windows, :jruby do
  gem "tzinfo", ">= 1", "< 3"
  gem "tzinfo-data"
end

# Performance-booster for watching directories on Windows
gem "wdm", "~> 0.2.0", :platforms => [:windows]
