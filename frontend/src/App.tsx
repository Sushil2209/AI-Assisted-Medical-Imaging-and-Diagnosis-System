import { useEffect, useRef, useState } from 'react'
import { getFirstInstanceId, getHealth, getPreviewUrl, getStudies, predict, type Prediction, type Study } from './api'
import './App.css'

const NAV_ITEMS = ['Dashboard', 'Imaging Studies', 'AI Analysis', 'Settings'] as const
type NavItem = (typeof NAV_ITEMS)[number]

function Sidebar({ active, onSelect }: { active: NavItem; onSelect: (item: NavItem) => void }) {
  return (
    <aside className="sidebar">
      <div className="brand">
        <span className="brand-mark" aria-hidden="true">N</span>
        <span className="brand-name">Nexvia</span>
      </div>
      <nav aria-label="Main navigation">
        <ul className="nav-list">
          {NAV_ITEMS.map((item) => (
            <li key={item}>
              <button
                type="button"
                className={item === active ? 'nav-item active' : 'nav-item'}
                aria-current={item === active ? 'page' : undefined}
                onClick={() => onSelect(item)}
              >
                {item}
              </button>
            </li>
          ))}
        </ul>
      </nav>
    </aside>
  )
}

type BackendStatus = 'checking' | 'connected' | 'disconnected'
const STATUS_LABEL: Record<BackendStatus, string> = {
  checking: 'Checking...',
  connected: 'Connected',
  disconnected: 'Not connected',
}

function Header({ status }: { status: BackendStatus }) {
  return (
    <header className="header">
      <div>
        <h1>Medical Imaging Dashboard</h1>
        <p className="subtitle">AI-assisted chest X-ray analysis</p>
      </div>
      <div className="status" role="status">
        <span
          className="status-dot"
          style={status === 'connected' ? { background: '#22c55e' } : status === 'disconnected' ? { background: '#ef4444' } : undefined}
          aria-hidden="true"
        />
        {STATUS_LABEL[status]}
      </div>
    </header>
  )
}

function Card({ title, className, children }: { title: string; className?: string; children: React.ReactNode }) {
  return (
    <section className={className ? `card ${className}` : 'card'}>
      <h2>{title}</h2>
      {children}
    </section>
  )
}

function App() {
  const [active, setActive] = useState<NavItem>('Dashboard')
  const [status, setStatus] = useState<BackendStatus>('checking')
  const [studies, setStudies] = useState<Study[] | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [selectedId, setSelectedId] = useState<string | null>(null)
  const [instanceId, setInstanceId] = useState<string | null>(null)
  const [instanceLoading, setInstanceLoading] = useState(false)
  const [instanceError, setInstanceError] = useState<string | null>(null)
  const [prediction, setPrediction] = useState<Prediction | null>(null)
  const [predicting, setPredicting] = useState(false)
  const [predictError, setPredictError] = useState<string | null>(null)
  const selectionRef = useRef<string | null>(null)

  async function selectStudy(id: string) {
    if (id === selectedId) return
    selectionRef.current = id
    setSelectedId(id)
    setInstanceId(null)
    setInstanceError(null)
    setPrediction(null)
    setPredictError(null)
    setPredicting(false)
    setInstanceLoading(true)
    try {
      const instance = await getFirstInstanceId(id)
      if (selectionRef.current === id) setInstanceId(instance)
    } catch (e) {
      if (selectionRef.current === id) setInstanceError(e instanceof Error ? e.message : 'Failed to load study instance')
    } finally {
      if (selectionRef.current === id) setInstanceLoading(false)
    }
  }

  async function runAnalysis() {
    if (!instanceId) return
    const id = selectionRef.current
    setPredicting(true)
    setPredictError(null)
    setPrediction(null)
    try {
      const result = await predict(instanceId)
      if (selectionRef.current === id) setPrediction(result)
    } catch (e) {
      if (selectionRef.current === id) setPredictError(e instanceof Error ? e.message : 'Prediction failed')
    } finally {
      if (selectionRef.current === id) setPredicting(false)
    }
  }

  useEffect(() => {
    getHealth()
      .then(() => setStatus('connected'))
      .catch(() => setStatus('disconnected'))
  }, [])

  async function loadStudies() {
    setLoading(true)
    setError(null)
    try {
      setStudies((await getStudies()).studies)
    } catch (e) {
      setStudies(null)
      setError(e instanceof Error ? e.message : 'Failed to load studies')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="layout">
      <Sidebar active={active} onSelect={setActive} />
      <div className="main">
        <Header status={status} />
        <main className="content">
          <Card title="Scan Viewer" className="viewer">
            {!selectedId && <div className="placeholder viewer-box">Select a study to view an image</div>}
            {selectedId && instanceLoading && <div className="placeholder viewer-box">Loading image...</div>}
            {selectedId && instanceError && (
              <div className="placeholder viewer-box" role="alert">Error: {instanceError}</div>
            )}
            {instanceId && (
              <div className="viewer-box viewer-image">
                <img src={getPreviewUrl(instanceId)} alt="Selected chest X-ray" />
              </div>
            )}
            <button
              type="button"
              className="primary-btn"
              onClick={runAnalysis}
              disabled={!instanceId || predicting}
            >
              {predicting ? 'Analyzing...' : 'Run AI Analysis'}
            </button>
          </Card>
          <Card title="AI Prediction">
            {predicting && <div className="placeholder">Running analysis...</div>}
            {predictError && <div className="placeholder" role="alert">Error: {predictError}</div>}
            {!predicting && !predictError && !prediction && <div className="placeholder">No prediction yet</div>}
            {prediction && (
              <div className="prediction">
                <p><strong>{prediction.predicted_class}</strong></p>
                <p className="muted">
                  Pneumonia probability: {(prediction.pneumonia_probability * 100).toFixed(2)}%
                </p>
              </div>
            )}
          </Card>
          <Card title="Grad-CAM">
            {prediction?.gradcam_png_base64 ? (
              <img
                className="gradcam-img"
                src={`data:image/png;base64,${prediction.gradcam_png_base64}`}
                alt="Grad-CAM heatmap"
              />
            ) : (
              <div className="placeholder">{predicting ? 'Generating heatmap...' : 'No heatmap available'}</div>
            )}
          </Card>
          <Card title="Studies" className="studies">
            {loading && <p className="muted">Loading studies...</p>}
            {error && <p className="muted" role="alert">Error: {error}</p>}
            {!loading && !error && studies === null && <p className="muted">No studies loaded.</p>}
            {!loading && studies !== null && studies.length === 0 && (
              <p className="muted">No studies found in Orthanc.</p>
            )}
            {!loading && studies !== null && studies.length > 0 && (
              <div className="table-wrap">
                <table className="studies-table">
                  <thead>
                    <tr>
                      <th>Study ID</th>
                      <th>Patient ID</th>
                      <th>Date</th>
                      <th>Description</th>
                    </tr>
                  </thead>
                  <tbody>
                    {studies.map((s) => (
                      <tr
                        key={s.id}
                        className={s.id === selectedId ? 'selected' : undefined}
                        aria-selected={s.id === selectedId}
                        tabIndex={0}
                        onClick={() => selectStudy(s.id)}
                        onKeyDown={(e) => {
                          if (e.key === 'Enter' || e.key === ' ') {
                            e.preventDefault()
                            selectStudy(s.id)
                          }
                        }}
                      >
                        <td>{s.id}</td>
                        <td>{s.patient_id ?? '—'}</td>
                        <td>{s.study_date ?? '—'}</td>
                        <td>{s.description ?? '—'}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
            <button type="button" className="primary-btn" onClick={loadStudies} disabled={loading}>
              {loading ? 'Loading...' : 'Load Studies'}
            </button>
          </Card>
        </main>
        <footer className="disclaimer">For educational use only. Not for clinical diagnosis.</footer>
      </div>
    </div>
  )
}

export default App
