import React, { useEffect, useMemo, useRef, useState } from "react";

const STORAGE_KEY = "dialogue-alternator-tool-v2";

function makeId() {
  if (typeof crypto !== "undefined" && crypto.randomUUID) return crypto.randomUUID();
  return `${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

function blankLine(speakerIndex = 0, type = "dialogue") {
  return {
    id: makeId(),
    type,
    speakerIndex,
    text: "",
  };
}

export default function App() {
  const [characterA, setCharacterA] = useState("Older waiter");
  const [characterB, setCharacterB] = useState("Younger waiter");
  const [firstSpeaker, setFirstSpeaker] = useState(0);
  const [includeNames, setIncludeNames] = useState(true);
  const [blankLineBetween, setBlankLineBetween] = useState(false);
  const [lines, setLines] = useState([blankLine(0)]);
  const textRefs = useRef({});

  useEffect(() => {
    try {
      const raw = localStorage.getItem(STORAGE_KEY);
      if (!raw) return;
      const saved = JSON.parse(raw);
      if (saved.characterA) setCharacterA(saved.characterA);
      if (saved.characterB) setCharacterB(saved.characterB);
      if (typeof saved.firstSpeaker === "number") setFirstSpeaker(saved.firstSpeaker);
      if (typeof saved.includeNames === "boolean") setIncludeNames(saved.includeNames);
      if (typeof saved.blankLineBetween === "boolean") setBlankLineBetween(saved.blankLineBetween);
      if (Array.isArray(saved.lines) && saved.lines.length) setLines(saved.lines);
    } catch {
      // Ignore corrupt local storage.
    }
  }, []);

  useEffect(() => {
    localStorage.setItem(
      STORAGE_KEY,
      JSON.stringify({ characterA, characterB, firstSpeaker, includeNames, blankLineBetween, lines })
    );
  }, [characterA, characterB, firstSpeaker, includeNames, blankLineBetween, lines]);

  const names = useMemo(
    () => [characterA || "Character A", characterB || "Character B"],
    [characterA, characterB]
  );

  function lastDialogueSpeakerIndex(currentLines = lines) {
    for (let i = currentLines.length - 1; i >= 0; i--) {
      if (currentLines[i].type === "dialogue") return currentLines[i].speakerIndex;
    }
    return firstSpeaker === 0 ? 1 : 0;
  }

  function addDialogueLine(afterId = null) {
    setLines((prev) => {
      const nextSpeaker = lastDialogueSpeakerIndex(prev) === 0 ? 1 : 0;
      const nextLine = blankLine(nextSpeaker, "dialogue");
      if (!afterId) return [...prev, nextLine];
      const idx = prev.findIndex((line) => line.id === afterId);
      if (idx < 0) return [...prev, nextLine];
      const copy = [...prev];
      copy.splice(idx + 1, 0, nextLine);
      setTimeout(() => textRefs.current[nextLine.id]?.focus(), 0);
      return copy;
    });
  }

  function addNarrationLine(afterId = null) {
    setLines((prev) => {
      const nextLine = blankLine(0, "narration");
      if (!afterId) return [...prev, nextLine];
      const idx = prev.findIndex((line) => line.id === afterId);
      if (idx < 0) return [...prev, nextLine];
      const copy = [...prev];
      copy.splice(idx + 1, 0, nextLine);
      setTimeout(() => textRefs.current[nextLine.id]?.focus(), 0);
      return copy;
    });
  }

  function updateLine(id, patch) {
    setLines((prev) => prev.map((line) => (line.id === id ? { ...line, ...patch } : line)));
  }

  function deleteLine(id) {
    setLines((prev) => (prev.length === 1 ? [blankLine(firstSpeaker)] : prev.filter((line) => line.id !== id)));
  }

  function clearDraft() {
    if (!window.confirm("Clear the current dialogue?")) return;
    setLines([blankLine(firstSpeaker)]);
  }

  function setStartingSpeaker(value) {
    setFirstSpeaker(value);
    setLines((prev) => {
      if (prev.length === 1 && prev[0].type === "dialogue" && !prev[0].text.trim()) {
        return [{ ...prev[0], speakerIndex: value }];
      }
      return prev;
    });
  }

  function realternate() {
    setLines((prev) => {
      let speaker = firstSpeaker;
      return prev.map((line) => {
        if (line.type !== "dialogue") return line;
        const patched = { ...line, speakerIndex: speaker };
        speaker = speaker === 0 ? 1 : 0;
        return patched;
      });
    });
  }

  function buildText() {
    const rendered = lines
      .map((line) => {
        const text = line.text.trimEnd();
        if (!text.trim()) return "";
        if (line.type === "narration") return text;
        return includeNames ? `${names[line.speakerIndex]}: ${text}` : text;
      })
      .filter(Boolean);

    return rendered.join(blankLineBetween ? "\n\n" : "\n");
  }

  async function copyText() {
    await navigator.clipboard.writeText(buildText());
  }

  function downloadTxt() {
    const text = buildText();
    const blob = new Blob([text], { type: "text/plain;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const safeA = (characterA || "dialogue").toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "");
    const safeB = (characterB || "draft").toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "");
    const link = document.createElement("a");
    link.href = url;
    link.download = `${safeA}-${safeB}-dialogue.txt`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  }

  function handleKeyDown(event, line) {
    if ((event.ctrlKey || event.metaKey) && event.shiftKey && event.key === "Enter") {
      event.preventDefault();
      addNarrationLine(line.id);
      return;
    }
    if ((event.ctrlKey || event.metaKey) && event.key === "Enter") {
      event.preventDefault();
      addDialogueLine(line.id);
    }
  }

  const filled = lines.filter((line) => line.text.trim()).length;
  const dialogue = lines.filter((line) => line.type === "dialogue" && line.text.trim()).length;
  const narration = lines.filter((line) => line.type === "narration" && line.text.trim()).length;

  return (
    <>
      <style>{CSS}</style>
      <main className="app">
        <header className="header">
          <div>
            <h1>Dialogue Alternator</h1>
            <p>Write back-and-forth dialogue without hand-coloring or losing track of whose line is whose.</p>
          </div>
          <div className="headerButtons">
            <button onClick={copyText}>Copy text</button>
            <button className="primary" onClick={downloadTxt}>Export .txt</button>
          </div>
        </header>

        <section className="layout">
          <aside className="panel">
            <h2>Setup</h2>

            <label>
              Character A
              <input value={characterA} onChange={(e) => setCharacterA(e.target.value)} />
            </label>

            <label>
              Character B
              <input value={characterB} onChange={(e) => setCharacterB(e.target.value)} />
            </label>

            <div className="fieldLabel">Who goes first?</div>
            <div className="choiceRow">
              {[0, 1].map((idx) => (
                <button
                  key={idx}
                  className={firstSpeaker === idx ? "selected" : ""}
                  onClick={() => setStartingSpeaker(idx)}
                >
                  {names[idx]}
                </button>
              ))}
            </div>

            <label className="check">
              <span>Include speaker names in export</span>
              <input type="checkbox" checked={includeNames} onChange={(e) => setIncludeNames(e.target.checked)} />
            </label>

            <label className="check">
              <span>Blank line between exported lines</span>
              <input type="checkbox" checked={blankLineBetween} onChange={(e) => setBlankLineBetween(e.target.checked)} />
            </label>

            <div className="stats">
              <div><strong>{filled}</strong><span>filled</span></div>
              <div><strong>{dialogue}</strong><span>dialogue</span></div>
              <div><strong>{narration}</strong><span>narration</span></div>
            </div>

            <div className="buttonStack">
              <button onClick={() => addDialogueLine()}>+ Add next dialogue line</button>
              <button onClick={() => addNarrationLine()}>+ Add narration line</button>
              <button onClick={realternate}>Re-alternate dialogue</button>
              <button className="danger" onClick={clearDraft}>Clear draft</button>
            </div>

            <div className="shortcuts">
              <strong>Shortcuts</strong>
              <p><kbd>Ctrl</kbd>/<kbd>⌘</kbd> + <kbd>Enter</kbd>: add dialogue below</p>
              <p><kbd>Ctrl</kbd>/<kbd>⌘</kbd> + <kbd>Shift</kbd> + <kbd>Enter</kbd>: add narration below</p>
            </div>
          </aside>

          <section className="editor">
            {lines.map((line, index) => {
              const isNarration = line.type === "narration";
              const cardClass = isNarration ? "line narration" : `line speaker${line.speakerIndex}`;
              return (
                <article key={line.id} className={cardClass}>
                  <div className="lineTop">
                    <div className="lineControlsLeft">
                      <span className="badge">{isNarration ? "Narration" : names[line.speakerIndex]}</span>
                      {!isNarration && (
                        <span className="speakerSwitch">
                          {[0, 1].map((idx) => (
                            <button
                              key={idx}
                              className={line.speakerIndex === idx ? "selected" : ""}
                              onClick={() => updateLine(line.id, { speakerIndex: idx })}
                            >
                              {names[idx]}
                            </button>
                          ))}
                        </span>
                      )}
                    </div>
                    <div className="lineControlsRight">
                      <button title="Insert dialogue below" onClick={() => addDialogueLine(line.id)}>+ Dialogue</button>
                      <button title="Insert narration below" onClick={() => addNarrationLine(line.id)}>+ Narration</button>
                      <button title="Delete line" onClick={() => deleteLine(line.id)}>Delete</button>
                    </div>
                  </div>

                  <textarea
                    ref={(el) => { textRefs.current[line.id] = el; }}
                    value={line.text}
                    onChange={(e) => updateLine(line.id, { text: e.target.value })}
                    onKeyDown={(e) => handleKeyDown(e, line)}
                    placeholder={isNarration ? "A descriptive sentence, stage direction, or pause..." : `${names[line.speakerIndex]}'s line...`}
                  />

                  <div className="lineFooter">
                    <span>Line {index + 1}</span>
                    <span>{line.text.length} chars</span>
                  </div>
                </article>
              );
            })}
          </section>
        </section>
      </main>
    </>
  );
}

const CSS = `
:root {
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  color: #171717;
  background: #f6f6f4;
}
* { box-sizing: border-box; }
body { margin: 0; }
button, input, textarea { font: inherit; }
button { cursor: pointer; }
.app {
  width: min(1180px, calc(100vw - 32px));
  margin: 0 auto;
  padding: 36px 0;
}
.header {
  display: flex;
  align-items: end;
  justify-content: space-between;
  gap: 24px;
  margin-bottom: 24px;
}
h1 { margin: 0; font-size: clamp(32px, 5vw, 52px); letter-spacing: -0.04em; }
p { color: #666; line-height: 1.5; }
.header p { margin: 8px 0 0; max-width: 680px; }
.headerButtons, .choiceRow, .buttonStack { display: flex; gap: 10px; flex-wrap: wrap; }
.headerButtons button, .panel button, .line button {
  border: 1px solid #d4d4d4;
  background: white;
  color: #171717;
  border-radius: 14px;
  padding: 10px 14px;
  transition: 120ms ease;
}
button:hover { transform: translateY(-1px); box-shadow: 0 8px 20px rgb(0 0 0 / 0.07); }
button.primary, button.selected {
  background: #171717;
  border-color: #171717;
  color: white;
}
button.danger { color: #b42318; }
.layout {
  display: grid;
  grid-template-columns: 340px 1fr;
  gap: 20px;
  align-items: start;
}
.panel, .line {
  background: white;
  border: 1px solid #e5e5e5;
  border-radius: 24px;
  box-shadow: 0 10px 30px rgb(0 0 0 / 0.06);
}
.panel {
  padding: 20px;
  position: sticky;
  top: 16px;
}
h2 { margin: 0 0 18px; }
label { display: block; margin-bottom: 14px; color: #555; font-size: 13px; font-weight: 650; text-transform: uppercase; letter-spacing: 0.04em; }
input[type="text"], input:not([type]), .panel input:not([type="checkbox"]) {
  display: block;
  width: 100%;
  margin-top: 6px;
  border: 1px solid #d4d4d4;
  border-radius: 14px;
  padding: 11px 12px;
  background: #fff;
  color: #171717;
  outline: none;
}
input:focus, textarea:focus { border-color: #171717; box-shadow: 0 0 0 3px rgb(23 23 23 / 0.12); }
.fieldLabel { margin: 18px 0 8px; color: #555; font-size: 13px; font-weight: 650; text-transform: uppercase; letter-spacing: 0.04em; }
.choiceRow { display: grid; grid-template-columns: 1fr 1fr; margin-bottom: 16px; }
.check {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  border: 1px solid #e5e5e5;
  border-radius: 18px;
  padding: 12px;
  text-transform: none;
  letter-spacing: 0;
  color: #333;
  background: #fafafa;
}
.check input { width: 18px; height: 18px; }
.stats {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 8px;
  margin: 16px 0;
}
.stats div {
  border: 1px solid #e5e5e5;
  border-radius: 18px;
  padding: 12px 8px;
  text-align: center;
  background: #fafafa;
}
.stats strong { display: block; font-size: 22px; }
.stats span { display: block; color: #777; font-size: 12px; }
.buttonStack { flex-direction: column; }
.buttonStack button { width: 100%; text-align: left; }
.shortcuts {
  margin-top: 16px;
  border-radius: 18px;
  background: #f4f4f4;
  padding: 14px;
  color: #666;
  font-size: 13px;
}
.shortcuts p { margin: 6px 0 0; }
kbd {
  display: inline-block;
  border: 1px solid #d4d4d4;
  border-bottom-width: 2px;
  border-radius: 6px;
  padding: 0 5px;
  background: white;
  color: #333;
}
.editor { display: grid; gap: 14px; }
.line { padding: 14px; }
.line.speaker0 { background: #eff6ff; border-color: #bfdbfe; }
.line.speaker1 { background: #fffbeb; border-color: #fde68a; }
.line.narration { background: white; border-color: #d4d4d4; }
.lineTop {
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 12px;
  flex-wrap: wrap;
  margin-bottom: 10px;
}
.lineControlsLeft, .lineControlsRight { display: flex; align-items: center; gap: 8px; flex-wrap: wrap; }
.badge {
  border-radius: 999px;
  padding: 7px 11px;
  background: rgb(255 255 255 / 0.85);
  border: 1px solid rgb(0 0 0 / 0.08);
  font-size: 13px;
  font-weight: 750;
}
.speakerSwitch {
  display: inline-flex;
  gap: 4px;
  background: white;
  border: 1px solid #e5e5e5;
  border-radius: 999px;
  padding: 4px;
}
.speakerSwitch button, .lineControlsRight button {
  padding: 7px 10px;
  border-radius: 999px;
  font-size: 12px;
}
textarea {
  display: block;
  width: 100%;
  min-height: 98px;
  resize: vertical;
  border: 1px solid rgb(0 0 0 / 0.08);
  border-radius: 18px;
  background: rgb(255 255 255 / 0.86);
  color: #171717;
  padding: 13px 14px;
  outline: none;
  line-height: 1.6;
  font-size: 17px;
}
.lineFooter {
  display: flex;
  justify-content: space-between;
  color: #777;
  font-size: 12px;
  margin-top: 8px;
}
@media (max-width: 860px) {
  .header { align-items: start; flex-direction: column; }
  .layout { grid-template-columns: 1fr; }
  .panel { position: static; }
}
`;

