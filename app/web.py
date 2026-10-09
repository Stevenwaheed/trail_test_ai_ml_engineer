"""
A minimal, ungraded testing UI.

This just gives you a quick way to poke at answer_question() from a browser
instead of writing your own throwaway script. It is NOT part of what's
graded -- it exists so `make up` gives you (and us, in Task 3) a fast way to
run the eval queries interactively.

Run it with `make up` (see the Makefile) or directly:

    uvicorn app.web:app --reload --port 5000

Then open http://localhost:5000
"""

import json
from html import escape
from pathlib import Path

from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse

from .users import TEST_USERS

app = FastAPI(title="Trial Test UI")

EVAL_QUERIES_PATH = Path(__file__).resolve().parent.parent / "eval" / "queries.json"


def _load_eval_queries() -> list:
    try:
        return json.loads(EVAL_QUERIES_PATH.read_text())
    except (OSError, ValueError):
        return []


def _user_options(selected: str) -> str:
    opts = []
    for name, groups in TEST_USERS.items():
        sel = " selected" if name == selected else ""
        opts.append(
            f'<option value="{escape(name)}"{sel}>{escape(name)} &middot; '
            f'{escape(", ".join(groups))}</option>'
        )
    return "\n".join(opts)


def _eval_chips() -> str:
    chips = []
    for q in _load_eval_queries():
        chips.append(
            f'<button type="button" class="chip" '
            f'data-user="{escape(q.get("user", ""))}" data-query="{escape(q.get("query", ""))}" '
            f'title="{escape(q.get("query", ""))}">'
            f'<span class="chip-id">{escape(q.get("id", ""))}</span>'
            f'<span class="chip-user">{escape(q.get("user", ""))}</span>'
            f'<span class="chip-text">{escape(q.get("query", ""))}</span></button>'
        )
    if not chips:
        return ""
    return f"""
    <div class="evals">
      <div class="section-label">Eval queries</div>
      <div class="chips">{"".join(chips)}</div>
    </div>
    """


def _badge(label: str, value) -> str:
    if value is True:
        cls, text = "yes", "yes"
    elif value is False:
        cls, text = "no", "no"
    else:
        cls, text = "unknown", escape(str(value))
    return f'<span class="badge {cls}"><span class="badge-label">{escape(label)}</span>{text}</span>'


def _citations_html(citations) -> str:
    if not citations:
        return '<span class="muted">None</span>'
    if not isinstance(citations, (list, tuple)):
        citations = [citations]
    items = "".join(
        f"<li><code>{escape(json.dumps(c) if isinstance(c, (dict, list)) else str(c))}</code></li>"
        for c in citations
    )
    return f'<ul class="citations">{items}</ul>'


PAGE_TEMPLATE = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Trial Test UI</title>
  <style>
    :root {{
      --bg: #f6f7f9;
      --surface: #ffffff;
      --text: #1a1d23;
      --muted: #6b7280;
      --border: #e3e6ea;
      --accent: #3b5bdb;
      --accent-hover: #2f4ac0;
      --accent-soft: #edf1ff;
      --ok: #1f7a4d;
      --ok-soft: #e6f5ec;
      --bad: #b42318;
      --bad-soft: #fdecea;
      --code-bg: #f1f3f5;
      --shadow: 0 1px 2px rgba(16, 24, 40, .05), 0 1px 3px rgba(16, 24, 40, .06);
    }}
    @media (prefers-color-scheme: dark) {{
      :root {{
        --bg: #0f1115;
        --surface: #171a21;
        --text: #e6e8eb;
        --muted: #9aa1ac;
        --border: #2a2f39;
        --accent: #7c93ff;
        --accent-hover: #95a8ff;
        --accent-soft: #1e2542;
        --ok: #5cc98f;
        --ok-soft: #13291e;
        --bad: #ff8a80;
        --bad-soft: #321a1a;
        --code-bg: #20242d;
        --shadow: none;
      }}
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font: 15px/1.55 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }}
    .wrap {{ max-width: 760px; margin: 0 auto; padding: 40px 16px 64px; }}
    header h1 {{ font-size: 22px; margin: 0 0 4px; letter-spacing: -.01em; }}
    header p {{ margin: 0 0 24px; color: var(--muted); }}
    code {{ font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: .9em;
            background: var(--code-bg); padding: 1px 5px; border-radius: 4px; }}
    .card {{
      background: var(--surface); border: 1px solid var(--border);
      border-radius: 12px; padding: 20px; box-shadow: var(--shadow);
    }}
    .section-label {{
      font-size: 12px; font-weight: 600; text-transform: uppercase;
      letter-spacing: .05em; color: var(--muted); margin-bottom: 8px;
    }}
    label {{ display: block; font-weight: 600; font-size: 13px; margin: 0 0 6px; }}
    .row {{ margin-bottom: 16px; }}
    select, textarea {{
      width: 100%; font: inherit; color: var(--text); background: var(--surface);
      border: 1px solid var(--border); border-radius: 8px; padding: 10px 12px;
      transition: border-color .15s, box-shadow .15s;
    }}
    textarea {{ resize: vertical; min-height: 84px; }}
    select:focus, textarea:focus {{
      outline: none; border-color: var(--accent);
      box-shadow: 0 0 0 3px var(--accent-soft);
    }}
    .actions {{ display: flex; align-items: center; gap: 12px; }}
    .hint {{ color: var(--muted); font-size: 13px; }}
    button.primary {{
      font: inherit; font-weight: 600; color: #fff; background: var(--accent);
      border: 0; border-radius: 8px; padding: 10px 18px; cursor: pointer;
      display: inline-flex; align-items: center; gap: 8px;
    }}
    button.primary:hover {{ background: var(--accent-hover); }}
    button.primary:disabled {{ opacity: .7; cursor: progress; }}
    .spinner {{
      width: 14px; height: 14px; border: 2px solid rgba(255,255,255,.4);
      border-top-color: #fff; border-radius: 50%; display: none;
      animation: spin .7s linear infinite;
    }}
    .loading .spinner {{ display: inline-block; }}
    @keyframes spin {{ to {{ transform: rotate(360deg); }} }}

    .evals {{ margin-top: 20px; }}
    .chips {{ display: grid; gap: 8px; }}
    .chip {{
      font: inherit; font-size: 13px; text-align: left; color: var(--text);
      background: var(--surface); border: 1px solid var(--border); border-radius: 8px;
      padding: 8px 10px; cursor: pointer; display: flex; gap: 8px; align-items: baseline;
      min-width: 0;
    }}
    .chip:hover {{ border-color: var(--accent); background: var(--accent-soft); }}
    .chip-id {{ font-weight: 700; color: var(--accent); }}
    .chip-user {{ color: var(--muted); font-family: ui-monospace, Menlo, monospace; font-size: 12px; }}
    .chip-text {{ overflow: hidden; text-overflow: ellipsis; white-space: nowrap; min-width: 0; }}

    .result {{ margin-top: 24px; }}
    .result-head {{ display: flex; flex-wrap: wrap; gap: 8px; align-items: center;
                    justify-content: space-between; margin-bottom: 14px; }}
    .asked {{ color: var(--muted); font-size: 13px; }}
    .badges {{ display: flex; gap: 8px; flex-wrap: wrap; }}
    .badge {{
      font-size: 12px; font-weight: 600; border-radius: 999px; padding: 3px 10px;
      border: 1px solid var(--border); display: inline-flex; gap: 6px;
    }}
    .badge-label {{ font-weight: 500; color: var(--muted); }}
    .badge.yes {{ background: var(--ok-soft); color: var(--ok); border-color: transparent; }}
    .badge.no {{ background: var(--bad-soft); color: var(--bad); border-color: transparent; }}
    .answer {{ white-space: pre-wrap; font-size: 16px; margin: 0 0 18px; }}
    .citations {{ margin: 0; padding-left: 18px; }}
    .citations li {{ margin: 2px 0; }}
    .muted {{ color: var(--muted); }}
    .error {{ border-color: var(--bad); background: var(--bad-soft); }}
    .error .section-label {{ color: var(--bad); }}
    .error pre {{ margin: 0; white-space: pre-wrap; word-break: break-word;
                  font-family: ui-monospace, Menlo, monospace; font-size: 13px; color: var(--bad); }}
  </style>
</head>
<body>
  <div class="wrap">
    <header>
      <h1>Trial Test UI</h1>
      <p>Pick a user, ask a question, and see what <code>answer_question()</code> returns.
         This page is just a harness &mdash; nothing here is graded.</p>
    </header>

    <form id="ask-form" class="card" method="post" action="/ask">
      <div class="row">
        <label for="user">User</label>
        <select id="user" name="user">
          {user_options}
        </select>
      </div>
      <div class="row">
        <label for="query">Question</label>
        <textarea id="query" name="query" rows="3" required
                  placeholder="e.g. What is the NOI for Oak Hill Plaza in Q3 2026?">{query}</textarea>
      </div>
      <div class="actions">
        <button id="submit" class="primary" type="submit"><span class="spinner"></span><span>Ask</span></button>
        <span class="hint">&#8984;/Ctrl + Enter to submit</span>
      </div>
    </form>

    {result}

    {eval_chips}
  </div>

  <script>
    (function () {{
      var form = document.getElementById("ask-form");
      var btn = document.getElementById("submit");
      var user = document.getElementById("user");
      var query = document.getElementById("query");

      form.addEventListener("submit", function () {{
        btn.disabled = true;
        btn.classList.add("loading");
        btn.lastElementChild.textContent = "Asking…";
      }});

      query.addEventListener("keydown", function (e) {{
        if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) {{
          e.preventDefault();
          form.requestSubmit();
        }}
      }});

      document.querySelectorAll(".chip").forEach(function (chip) {{
        chip.addEventListener("click", function () {{
          user.value = chip.dataset.user;
          query.value = chip.dataset.query;
          query.focus();
        }});
      }});
    }})();
  </script>
</body>
</html>
"""


def _render(query: str = "", user: str = "", result_html: str = "") -> str:
    return PAGE_TEMPLATE.format(
        user_options=_user_options(user),
        query=escape(query),
        result=result_html,
        eval_chips=_eval_chips(),
    )


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return _render()


@app.post("/ask", response_class=HTMLResponse)
async def ask(user: str = Form(...), query: str = Form(...)) -> str:
    groups = TEST_USERS.get(user, [])

    try:
        # Imported here (rather than at module load) so the UI still starts
        # even before you've implemented answer_question().
        from app.generate import answer_question

        result = await answer_question(query, groups)
        result_html = f"""
        <section class="card result">
          <div class="result-head">
            <span class="asked">{escape(user)} &middot; {escape(", ".join(groups) or "no groups")}</span>
            <div class="badges">
              {_badge("grounded", result.get("grounded"))}
              {_badge("used tool", result.get("used_tool"))}
            </div>
          </div>
          <div class="section-label">Answer</div>
          <p class="answer">{escape(str(result.get("answer", "")))}</p>
          <div class="section-label">Citations</div>
          {_citations_html(result.get("citations"))}
        </section>
        """
    except NotImplementedError as e:
        result_html = f"""
        <section class="card result error">
          <div class="section-label">Not implemented yet</div>
          <pre>{escape(str(e))}</pre>
        </section>
        """
    except Exception as e:  # noqa: BLE001 -- deliberately broad for a dev harness
        result_html = f"""
        <section class="card result error">
          <div class="section-label">{escape(type(e).__name__)}</div>
          <pre>{escape(str(e))}</pre>
        </section>
        """

    return _render(query=query, user=user, result_html=result_html)
