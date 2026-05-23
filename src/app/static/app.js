const page = document.body.dataset.page;

const fmt = new Intl.NumberFormat("en", { maximumFractionDigits: 2 });

function text(id, value) {
  const node = document.getElementById(id);
  if (node) node.textContent = value;
}

function badge(status) {
  const safe = status || "missing";
  return `<span class="badge ${safe}">${safe}</span>`;
}

async function getJson(url) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`${url} returned ${response.status}`);
  return response.json();
}

function renderRadar(flights) {
  const svg = document.getElementById("radarPlot");
  if (!svg) return;
  svg.innerHTML = "";
  const bg = document.createElementNS("http://www.w3.org/2000/svg", "rect");
  bg.setAttribute("class", "radar-bg");
  bg.setAttribute("x", "0");
  bg.setAttribute("y", "0");
  bg.setAttribute("width", "900");
  bg.setAttribute("height", "520");
  svg.appendChild(bg);

  for (let x = 80; x < 900; x += 120) {
    const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
    line.setAttribute("class", "grid-line");
    line.setAttribute("x1", x);
    line.setAttribute("x2", x);
    line.setAttribute("y1", "0");
    line.setAttribute("y2", "520");
    svg.appendChild(line);
  }

  for (let y = 80; y < 520; y += 110) {
    const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
    line.setAttribute("class", "grid-line");
    line.setAttribute("x1", "0");
    line.setAttribute("x2", "900");
    line.setAttribute("y1", y);
    line.setAttribute("y2", y);
    svg.appendChild(line);
  }

  const valid = flights.filter((f) => Number.isFinite(f.longitude) && Number.isFinite(f.latitude));
  if (!valid.length) return;
  const lons = valid.map((f) => f.longitude);
  const lats = valid.map((f) => f.latitude);
  const minLon = Math.min(...lons);
  const maxLon = Math.max(...lons);
  const minLat = Math.min(...lats);
  const maxLat = Math.max(...lats);

  for (const flight of valid) {
    const x = 40 + ((flight.longitude - minLon) / Math.max(maxLon - minLon, 0.001)) * 820;
    const y = 480 - ((flight.latitude - minLat) / Math.max(maxLat - minLat, 0.001)) * 440;
    const circle = document.createElementNS("http://www.w3.org/2000/svg", "circle");
    circle.setAttribute("class", `plane ${flight.status}`);
    circle.setAttribute("cx", x.toFixed(1));
    circle.setAttribute("cy", y.toFixed(1));
    circle.setAttribute("r", flight.status === "watch" || flight.status === "low-altitude" ? "7" : "5");
    circle.appendChild(document.createElementNS("http://www.w3.org/2000/svg", "title")).textContent =
      `${flight.callsign} ${flight.status}`;
    svg.appendChild(circle);
  }
}

function renderFlights(data) {
  const flights = data.flights || [];
  text("snapshotName", data.snapshot || "No snapshot");
  text("flightCount", `${data.count || 0} aircraft`);
  text("radarStatus", flights.length ? "Operational" : "No data");
  renderRadar(flights);

  const watch = flights.filter((f) => f.status !== "normal" && f.status !== "ground");
  text("watchCount", `${watch.length} watch`);
  const watchList = document.getElementById("watchList");
  if (watchList) {
    watchList.innerHTML = watch.slice(0, 12).map((f) => `
      <div class="watch-item">
        <strong>${f.callsign || f.icao24}</strong>
        ${badge(f.status)}
        <span>${f.origin_country || "Unknown"} · ${fmt.format(f.velocity || 0)} m/s · ${fmt.format(f.baro_altitude || 0)} m</span>
      </div>
    `).join("") || `<div class="watch-item"><strong>No active attention items</strong><span>All visible aircraft look normal.</span></div>`;
  }

  const table = document.getElementById("flightTable");
  if (table) {
    table.innerHTML = flights.slice(0, 80).map((f) => `
      <tr>
        <td>${f.callsign || f.icao24}</td>
        <td>${f.origin_country || "Unknown"}</td>
        <td>${badge(f.status)}</td>
        <td>${fmt.format(f.baro_altitude || 0)} m</td>
        <td>${fmt.format(f.velocity || 0)} m/s</td>
        <td>${fmt.format(f.vertical_rate || 0)} m/s</td>
      </tr>
    `).join("");
  }
}

function renderAdmin(summary) {
  const monitoring = summary.monitoring || {};
  const validation = summary.validation || {};
  const drift = summary.drift || {};
  const training = summary.training || {};
  const models = training.models || {};
  const trajectory = models.trajectory_lstm || {};
  const ground = models.on_ground_lstm || {};

  text("overallStatus", monitoring.status || "missing");
  text("validationStatus", validation.success === true || validation.status === "passed" ? "Passed" : "Check");
  text("validationDetail", validation.report_path || "No validation report loaded");
  text("driftStatus", drift.status || "missing");
  text("driftDetail", drift.gate_reason || "No drift summary loaded");
  text("trajectoryMetric", `${fmt.format(trajectory.mae_latitude || 0)} / ${fmt.format(trajectory.mae_longitude || 0)}`);
  text("groundMetric", `${fmt.format((ground.accuracy || 0) * 100)}% · F1 ${fmt.format(ground.f1 || 0)}`);

  const checks = monitoring.checks || [];
  text("checkCount", `${checks.length} checks`);
  const checkList = document.getElementById("monitorChecks");
  if (checkList) {
    checkList.innerHTML = checks.map((check) => `
      <div class="check-item">
        <strong>${check.name}</strong>
        ${badge(check.status)}
        <span>${check.message}</span>
      </div>
    `).join("") || `<div class="check-item"><strong>No monitoring report</strong><span>Run the monitor stage first.</span></div>`;
  }

  const inventory = document.getElementById("modelInventory");
  if (inventory) {
    inventory.innerHTML = (summary.models || []).map((model) => `
      <div class="check-item">
        <strong>${model.name}</strong>
        <span>${fmt.format(model.size_bytes || 0)} bytes</span>
      </div>
    `).join("") || `<div class="check-item"><strong>No model artifacts</strong><span>Train models first.</span></div>`;
  }
}

if (page === "ops") {
  getJson("/api/flights").then(renderFlights).catch((error) => text("radarStatus", error.message));
}

if (page === "admin") {
  getJson("/api/admin/summary").then(renderAdmin).catch((error) => text("overallStatus", error.message));
}
