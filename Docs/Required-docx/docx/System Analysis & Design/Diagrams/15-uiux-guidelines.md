# 15. UI/UX Guidelines

## 15.1 Design Principles
1. **Clarity first** — every screen answers one question.
2. **Progressive disclosure** — show essential KPIs, hide details behind expanders.
3. **Consistency** — same control placement (sidebar = filters, top = actions).
4. **Feedback** — loading spinners, success toasts, error banners on every long action.
5. **Reproducibility** — every chart can be downloaded as PNG, every table as CSV.
6. **Accessibility** — WCAG 2.1 AA: contrast ≥ 4.5 : 1, keyboard-navigable, alt-text on charts.

## 15.2 Color Palette

| Token              | Hex       | Usage                              |
|--------------------|-----------|------------------------------------|
| `--brand-primary`  | `#2C6FBB` | Buttons, headings, links            |
| `--brand-accent`   | `#F39C12` | Highlights, KPI deltas (positive)   |
| `--success`        | `#2EAA63` | Success toast / "in-control" status |
| `--warning`        | `#E0A106` | Drift warnings                      |
| `--danger`         | `#D94545` | Errors, threshold breaches          |
| `--bg-light`       | `#FAFCFF` | Page background                     |
| `--bg-panel`       | `#FFFFFF` | Cards, panels                       |
| `--text-primary`   | `#1A1F2B` | Body text                           |
| `--text-secondary` | `#5C6776` | Subtitles, helper text              |

Streamlit theme (`.streamlit/config.toml`):

```toml
[theme]
primaryColor       = "#2C6FBB"
backgroundColor    = "#FAFCFF"
secondaryBackgroundColor = "#FFFFFF"
textColor          = "#1A1F2B"
font               = "sans serif"
```

## 15.3 Typography
- Family: **Inter**, fallback `sans-serif`.
- Scale: 32 / 24 / 20 / 16 / 14 / 12 px.
- Numeric KPIs: tabular numerals (`font-variant-numeric: tabular-nums`).

## 15.4 Layout & Spacing
- 8-px base grid (`4 / 8 / 12 / 16 / 24 / 32 / 48`).
- Max content width: 1280 px, centered.
- Sidebar width: 300 px (collapsible).
- Cards: 12 px border-radius, 1 px `#E1E7F0` border, soft shadow `0 1px 3px rgba(0,0,0,.04)`.

## 15.5 Charts
- Library: **Plotly** (interactive, exportable, accessible).
- Forecast lines: 2 px solid; CI band: 20 % opacity fill.
- Always label axes and units. Use ISO date format on x-axis.
- Color-blind safe palette (Okabe-Ito) for multi-series.

## 15.6 Interactions & Feedback
| Event              | UI feedback                              |
|--------------------|------------------------------------------|
| Long compute (>1s) | `st.spinner("Generating forecast...")`   |
| Success            | `st.toast("Forecast ready ✅", icon="✅")` |
| Validation error   | `st.error()` banner + field highlight     |
| Destructive action | Modal confirmation (`st.modal`)           |

## 15.7 Accessibility Checklist
- [x] Contrast ratio ≥ 4.5:1 for text, ≥ 3:1 for UI components.
- [x] All interactive elements reachable via Tab.
- [x] Charts include `alt` description summarizing trend.
- [x] Avoid color-only signals — pair with icon / text.
- [x] Form labels explicit; never placeholder-only.
- [x] Respect `prefers-reduced-motion` for animations.

## 15.8 Empty / Error / Loading states
Every page provides three states:
1. **Empty** — illustration + CTA ("Upload data to begin").
2. **Loading** — skeleton placeholder + spinner.
3. **Error** — friendly message + "Retry" + link to logs.

