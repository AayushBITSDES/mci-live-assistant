import { createServer } from "node:http";

const PORT = Number(process.env.PORT ?? 8787);

const scenarios = {
  "b2-shanta-blood-test": {
    sentence: "You have a blood test at 10am, and it is time to leave now.",
    threshold: 0.7
  },
  "b1-sanjay-client-deck": {
    sentence: "Send the client deck now so the client has it before the 3pm meeting.",
    threshold: 0.85
  }
};

createServer(async (req, res) => {
  setCors(res);

  if (req.method === "OPTIONS") {
    res.writeHead(204);
    res.end();
    return;
  }

  if (req.method === "GET" && req.url === "/health") {
    sendJson(res, 200, { ok: true });
    return;
  }

  if (req.method === "POST" && req.url === "/api/scenario/event") {
    const body = await readJson(req);
    const scenario = scenarios[body.scenarioId] ?? scenarios["b2-shanta-blood-test"];
    const shouldNudge = Boolean(body.event === "risk_window" && Number(body.riskScore ?? 0) >= scenario.threshold);

    sendJson(res, 200, {
      shouldNudge,
      sentence: shouldNudge ? scenario.sentence : "",
      priority: Number(body.riskScore ?? 0) > 0.9 ? "high" : "medium",
      reason: shouldNudge ? "Deterministic prototype threshold crossed." : "Signal below deterministic threshold."
    });
    return;
  }

  if (req.method === "POST" && req.url === "/api/nudge/rewrite") {
    const body = await readJson(req);
    const scenario = scenarios[body.scenarioId] ?? scenarios["b2-shanta-blood-test"];

    sendJson(res, 200, {
      sentence: oneSentence(body.sentence || scenario.sentence),
      source: process.env.GEMINI_API_KEY ? "gemini-adapter-placeholder" : "deterministic"
    });
    return;
  }

  sendJson(res, 404, { error: "Not found" });
}).listen(PORT, () => {
  console.log(`Prototype API listening on http://localhost:${PORT}`);
});

function setCors(res) {
  res.setHeader("Access-Control-Allow-Origin", "*");
  res.setHeader("Access-Control-Allow-Methods", "GET,POST,OPTIONS");
  res.setHeader("Access-Control-Allow-Headers", "Content-Type");
}

function sendJson(res, status, data) {
  res.writeHead(status, { "Content-Type": "application/json" });
  res.end(JSON.stringify(data));
}

async function readJson(req) {
  const chunks = [];
  for await (const chunk of req) {
    chunks.push(chunk);
  }

  if (!chunks.length) {
    return {};
  }

  try {
    return JSON.parse(Buffer.concat(chunks).toString("utf8"));
  } catch {
    return {};
  }
}

function oneSentence(value) {
  const text = String(value).trim();
  const match = text.match(/^.*?[.!?](?:\s|$)/);
  return match ? match[0].trim() : `${text.replace(/[.!?]+$/g, "")}.`;
}
