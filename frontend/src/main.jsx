import React, { useEffect, useState, startTransition } from "react";
import { createRoot } from "react-dom/client";
import {
  Activity,
  Brain,
  ClipboardCheck,
  Database,
  Gauge,
  GitBranch,
  Layers,
  MapPinned,
  Plane,
  Radar,
  RefreshCw,
  ShieldCheck,
  Sparkles,
  TerminalSquare
} from "lucide-react";
import "./styles.css";

const number = new Intl.NumberFormat("en", { maximumFractionDigits: 2 });

function App() {
  const [briefing, setBriefing] = useState(null);
  const [admin, setAdmin] = useState(null);
  const [advanced, setAdvanced] = useState(null);
  const [selectedAircraft, setSelectedAircraft] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [predictionState, setPredictionState] = useState("idle");
  const [error, setError] = useState(null);

  useEffect(() => {
    let cancelled = false;
    async function load() {
      try {
        const [briefingResponse, adminResponse, advancedResponse] = await Promise.all([
          fetch("/api/intelligence/briefing"),
          fetch("/api/admin/summary"),
          fetch("/api/admin/advanced")
        ]);
        if (!briefingResponse.ok || !adminResponse.ok || !advancedResponse.ok) {
          throw new Error("SkyWatch API is not ready.");
        }
        const [briefingData, adminData, advancedData] = await Promise.all([
          briefingResponse.json(),
          adminResponse.json(),
          advancedResponse.json()
        ]);
        if (!cancelled) {
          startTransition(() => {
            setBriefing(briefingData);
            setAdmin(adminData);
            setAdvanced(advancedData);
            setSelectedAircraft(briefingData.attention_queue?.[0] || briefingData.traffic?.[0] || null);
          });
        }
      } catch (loadError) {
        if (!cancelled) setError(loadError.message);
      }
    }
    load();
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    if (!selectedAircraft?.icao24) return;
    let cancelled = false;
    async function loadPrediction() {
      setPredictionState("loading");
      setPrediction(null);
      try {
        const response = await fetch(`/api/predictions/${selectedAircraft.icao24}`);
        const data = await response.json();
        if (!response.ok) throw new Error(data.detail || "Prediction failed.");
        if (!cancelled) {
          setPrediction(data);
          setPredictionState("ready");
        }
      } catch (predictionError) {
        if (!cancelled) {
          setPredictionState(predictionError.message);
        }
      }
    }
    loadPrediction();
    return () => {
      cancelled = true;
    };
  }, [selectedAircraft]);

  if (error) {
    return <Shell><EmptyState title="API unavailable" message={error} /></Shell>;
  }

  async function updateModelStage(modelKey, stage) {
    try {
      const response = await fetch(`/api/admin/model-registry/${modelKey}/stage`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ stage, note: `Promoted from SkyWatch admin UI to ${stage}.` })
      });
      if (!response.ok) {
        const data = await response.json();
        throw new Error(data.detail || "Stage update failed.");
      }
      const advancedResponse = await fetch("/api/admin/advanced");
      setAdvanced(await advancedResponse.json());
    } catch (stageError) {
      setError(stageError.message);
    }
  }

  if (!briefing || !admin || !advanced) {
    return <Shell><EmptyState title="Loading SkyWatch intelligence" message="Preparing airspace briefing..." /></Shell>;
  }

  return (
    <Shell>
      <Hero summary={briefing.summary} snapshot={briefing.snapshot} />
      <InsightBar briefing={briefing} admin={admin} />
      <main className="workbench">
        <section className="panel priority-panel">
          <PanelTitle icon={<Radar />} title="Priority queue" detail="Ranked by operational signal" />
          <AircraftList
            aircraft={briefing.attention_queue.length ? briefing.attention_queue : briefing.traffic.slice(0, 10)}
            selected={selectedAircraft}
            onSelect={setSelectedAircraft}
          />
        </section>
        <section className="panel prediction-panel">
          <PanelTitle icon={<Brain />} title="Model copilot" detail="Live inference from trained models" />
          <PredictionCard aircraft={selectedAircraft} prediction={prediction} state={predictionState} />
        </section>
      </main>
      <AdminSnapshot admin={admin} />
      <AdvancedAdmin advanced={advanced} onStageChange={updateModelStage} />
    </Shell>
  );
}

function Shell({ children }) {
  return (
    <>
      <header className="topbar">
        <a className="brand" href="/">
          <Sparkles size={20} />
          SkyWatch Copilot
        </a>
        <nav>
          <a href="#copilot">Copilot</a>
          <a href="#admin">Admin</a>
          <a href="/docs">API docs</a>
        </nav>
      </header>
      <div className="shell" id="copilot">{children}</div>
    </>
  );
}

function Hero({ summary, snapshot }) {
  return (
    <section className={`hero ${summary.priority}`}>
      <div>
        <p className="eyebrow">Intelligent airspace experience</p>
        <h1>{summary.priority.replace("-", " ")} briefing</h1>
        <p className="briefing">{summary.briefing}</p>
      </div>
      <div className="hero-metrics">
        <Metric label="Snapshot" value={snapshot || "missing"} />
        <Metric label="Aircraft" value={summary.aircraft_count} />
        <Metric label="Attention" value={summary.attention_count} />
      </div>
    </section>
  );
}

function InsightBar({ briefing, admin }) {
  const readiness = briefing.model_readiness || {};
  const monitoring = admin.monitoring || {};
  return (
    <section className="insight-grid">
      <InfoTile icon={<Gauge />} label="Model readiness" value={readiness.label || "unknown"}>
        Trajectory MAE {format(readiness.trajectory_mae)} and on-ground accuracy {formatPercent(readiness.on_ground_accuracy)}.
      </InfoTile>
      <InfoTile icon={<ShieldCheck />} label="Production state" value={monitoring.status || "missing"}>
        {monitoring.checks?.find((check) => check.status === "warn")?.message || "Production checks are available."}
      </InfoTile>
      <InfoTile icon={<Activity />} label="Experience mode" value="human-in-the-loop">
        The UI explains why a flight matters before asking the user to inspect raw telemetry.
      </InfoTile>
    </section>
  );
}

function AircraftList({ aircraft, selected, onSelect }) {
  return (
    <div className="aircraft-list">
      {aircraft.map((item) => (
        <button
          key={item.icao24}
          className={`aircraft-card ${selected?.icao24 === item.icao24 ? "selected" : ""}`}
          onClick={() => onSelect(item)}
        >
          <span className={`status ${item.status}`}>{item.status}</span>
          <strong>{item.callsign || item.icao24}</strong>
          <small>{item.reason}</small>
          <span className="score">{item.attention_score}/100</span>
        </button>
      ))}
    </div>
  );
}

function PredictionCard({ aircraft, prediction, state }) {
  if (!aircraft) {
    return <EmptyState title="No aircraft selected" message="Select an aircraft from the priority queue." />;
  }
  const forecast = aircraft.forecast || {};
  const modelPrediction = prediction?.prediction || {};
  return (
    <article className="prediction-card">
      <div className="prediction-header">
        <Plane />
        <div>
          <span>Selected aircraft</span>
          <h2>{aircraft.callsign || aircraft.icao24}</h2>
        </div>
      </div>
      <div className="prediction-grid">
        <Metric label="Heuristic next altitude" value={forecast.altitude_m == null ? "unknown" : `${format(forecast.altitude_m)} m`} />
        <Metric label="Model next latitude" value={modelPrediction.next_latitude == null ? "loading" : format(modelPrediction.next_latitude)} />
        <Metric label="Model next longitude" value={modelPrediction.next_longitude == null ? "loading" : format(modelPrediction.next_longitude)} />
        <Metric label="On-ground probability" value={modelPrediction.next_on_ground_probability == null ? "loading" : formatPercent(modelPrediction.next_on_ground_probability)} />
      </div>
      <div className="recommendation">
        <h3>Recommended action</h3>
        <p>{prediction?.decision_support?.recommended_action || aircraft.recommended_action}</p>
        <small>{state === "ready" ? "Prediction loaded from trained Keras models." : state}</small>
      </div>
    </article>
  );
}

function AdminSnapshot({ admin }) {
  const training = admin.training?.models || {};
  const trajectory = training.trajectory_lstm || {};
  const ground = training.on_ground_lstm || {};
  return (
    <section className="admin-section panel" id="admin">
      <PanelTitle icon={<TerminalSquare />} title="Admin intelligence snapshot" detail="Validation, drift and model state" />
      <div className="admin-grid">
        <Metric label="Great Expectations" value={admin.validation?.status || "missing"} />
        <Metric label="Evidently drift" value={admin.drift?.status || "missing"} />
        <Metric label="Trajectory MAE" value={`${format(trajectory.mae_latitude)} / ${format(trajectory.mae_longitude)}`} />
        <Metric label="On-ground accuracy" value={formatPercent(ground.accuracy)} />
      </div>
      <div className="artifact-row">
        {(admin.models || []).map((model) => (
          <span key={model.name}>
            <MapPinned size={14} />
            {model.name}
          </span>
        ))}
      </div>
    </section>
  );
}

function AdvancedAdmin({ advanced, onStageChange }) {
  return (
    <section className="advanced-admin panel">
      <PanelTitle icon={<Layers />} title="Advanced admin console" detail="Quality, registry and lifecycle control" />
      <div className="quality-grid">
        {(advanced.quality_gates || []).map((gate) => (
          <QualityGate key={gate.name} gate={gate} />
        ))}
      </div>
      <div className="admin-deep-grid">
        <article className="deep-card experiment-card">
          <h3><GitBranch /> MLflow experiment tracking</h3>
          <dl>
            <div>
              <dt>Experiment</dt>
              <dd>{advanced.experiment_tracking?.experiment_name || "missing"}</dd>
            </div>
            <div>
              <dt>Tracking URI</dt>
              <dd>{advanced.experiment_tracking?.tracking_uri || "missing"}</dd>
            </div>
            <div>
              <dt>Registry</dt>
              <dd>{advanced.experiment_tracking?.registry_enabled ? "enabled" : "waiting for next run"}</dd>
            </div>
          </dl>
          <div className="metric-cloud">
            {(advanced.experiment_tracking?.metrics_logged || []).slice(0, 8).map((metric) => (
              <span key={metric}>{metric}</span>
            ))}
          </div>
        </article>
        <article className="deep-card reports-card">
          <h3><Database /> Evidence reports</h3>
          <div className="report-list">
            {(advanced.report_links || []).map((report) => (
              report.url && report.available ? (
                <a key={report.name} href={report.url} target="_blank" rel="noreferrer">
                  <ClipboardCheck size={16} />
                  {report.name}
                </a>
              ) : (
                <span key={report.name} className={report.available ? "available" : "missing"}>
                  <ClipboardCheck size={16} />
                  {report.name}: {report.available ? "available" : "missing"}
                </span>
              )
            ))}
          </div>
        </article>
      </div>
      <ModelRegistryBoard
        models={advanced.model_registry || []}
        stages={advanced.lifecycle_stages || []}
        onStageChange={onStageChange}
      />
    </section>
  );
}

function QualityGate({ gate }) {
  return (
    <article className={`quality-gate ${gate.status}`}>
      <span>{gate.status}</span>
      <h3>{gate.name}</h3>
      <p>{gate.message}</p>
    </article>
  );
}

function ModelRegistryBoard({ models, stages, onStageChange }) {
  return (
    <article className="registry-board">
      <div className="registry-heading">
        <h3><RefreshCw /> Model lifecycle registry</h3>
        <p>Local governance layer over MLflow registered models for staged promotion decisions.</p>
      </div>
      <div className="registry-list">
        {models.map((model) => (
          <div className="registry-item" key={model.key}>
            <div>
              <span className="registry-key">{model.key}</span>
              <h4>{model.registered_name}</h4>
              <p>{model.task}</p>
              <small>
                Version {model.registered_model_version || "pending"} · {model.model_uri || "registered after next training run"}
              </small>
            </div>
            <div className="stage-controls">
              <strong>{model.stage}</strong>
              <div>
                {stages.map((stage) => (
                  <button
                    key={stage}
                    className={stage === model.stage ? "active" : ""}
                    onClick={() => onStageChange(model.key, stage)}
                  >
                    {stage}
                  </button>
                ))}
              </div>
            </div>
          </div>
        ))}
      </div>
    </article>
  );
}

function PanelTitle({ icon, title, detail }) {
  return (
    <div className="panel-title">
      <h2>{icon}{title}</h2>
      <span>{detail}</span>
    </div>
  );
}

function InfoTile({ icon, label, value, children }) {
  return (
    <article className="info-tile">
      <span>{icon}{label}</span>
      <strong>{value}</strong>
      <p>{children}</p>
    </article>
  );
}

function Metric({ label, value }) {
  return (
    <div className="metric">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function EmptyState({ title, message }) {
  return (
    <section className="empty">
      <h1>{title}</h1>
      <p>{message}</p>
    </section>
  );
}

function format(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "n/a";
  return number.format(Number(value));
}

function formatPercent(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "n/a";
  return `${number.format(Number(value) * 100)}%`;
}

createRoot(document.getElementById("root")).render(<App />);
