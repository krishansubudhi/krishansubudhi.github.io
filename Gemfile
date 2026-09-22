source "https://rubygems.org"

# This site is published by the *default* GitHub Pages builder. That builder
# ignores this Gemfile and its lockfile entirely and builds server-side with its
# own `github-pages` gem set; these files exist for local builds and for the
# dependency scanner. So the Jekyll pin below is deliberately set to the exact
# version github-pages currently ships (jekyll 3.10.0), which keeps local output
# matching production without dragging in the ~65 extra gems the meta-gem pulls
# for features this site does not use (remote themes, GitHub API clients,
# CommonMark, html-pipeline). Those extras were the entire source of the
# dependency alerts; see the note at the bottom of this file.
#
# Run `bundle update` to pick up patch releases. If you ever need a Jekyll 4.x
# feature, be aware it requires switching publishing to a GitHub Actions
# workflow, because the default Pages builder does not support Jekyll 4.
gem "jekyll", "~> 3.10.0"

# This is the default theme for new Jekyll sites. You may change this to anything you like.
gem "minima", "~> 2.0"

# If you have any plugins, put them here!
group :jekyll_plugins do
  gem "jekyll-feed", "~> 0.6"
end

# Windows does not include zoneinfo files, so bundle the tzinfo-data gem
gem "tzinfo-data", platforms: [:mingw, :mswin, :x64_mingw, :jruby]

# Performance-booster for watching directories on Windows
gem "wdm", "~> 0.1.0" if Gem.win_platform?

gem 'jekyll-seo-tag'

gem "jekyll-sitemap", "~> 1.4"

# Jekyll defaults kramdown's input to GFM, and kramdown 2.x moved that parser
# out into its own gem. kramdown 1.x bundled it, so this was an implicit
# dependency before and has to be explicit now or the build dies on the first
# post. Same version the github-pages gem set pins.
gem "kramdown-parser-gfm", "~> 1.1"

# Why not `gem "github-pages"`? It was tried. Because it hard-depends on
# jekyll-remote-theme (which caps rubyzip < 3.0) and html-pipeline (nokogiri),
# resolving through the meta-gem reintroduces more advisories than it clears.
# The direct pins above resolve to a clean dependency tree on the same Jekyll.
