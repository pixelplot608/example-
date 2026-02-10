import './style.css'
import { supabase } from './supabase'

const API_BASE = 'http://localhost:8000'

type AuthUser = { user_id: string; email: string; role: string }
let currentUser: AuthUser | null = null

async function getAccessToken(): Promise<string | null> {
  const { data } = await supabase.auth.getSession()
  return data.session?.access_token ?? null
}

async function fetchWithAuth(url: string, options?: RequestInit): Promise<Response> {
  const token = await getAccessToken()
  const headers = new Headers(options?.headers)
  if (token) headers.set('Authorization', `Bearer ${token}`)
  return fetch(url, { ...options, headers })
}

type Region = {
  id: string
  name: string
  country_code: string
  population: number
}

type Facility = {
  id: string
  name: string
  facility_type: string
  region_id: string
  latitude: number
  longitude: number
  bed_capacity: number
}

type AccessMetric = {
  region_id: string
  region_name: string
  population: number
  centroid_lat: number
  centroid_lon: number
  distance_km: number
  access_score: number
  nearest_facility_id: string
  nearest_facility_name: string
}

declare global {
  interface Window {
    L: typeof import('leaflet')
    Chart: any
  }
}

const app = document.querySelector<HTMLDivElement>('#app')
let mapInstance: ReturnType<Window['L']['map']> | null = null
let mapLayers: { remove(): void }[] = []
let demandInitialized = false
let demandChart: any | null = null
let planningMap: ReturnType<Window['L']['map']> | null = null
let planningLayers: { remove(): void }[] = []
let mobileMap: ReturnType<Window['L']['map']> | null = null
let mobileLayers: { remove(): void }[] = []

async function fetchJson<T>(url: string, options?: RequestInit): Promise<T | null> {
  try {
    const res = await fetchWithAuth(url, options)
    if (!res.ok) return null
    return (await res.json()) as T
  } catch {
    return null
  }
}

function accessScoreToColor(score: number): string {
  const hue = Math.round(120 * score)
  return `hsl(${hue}, 70%, 42%)`
}

type ViewName = 'dashboard' | 'map' | 'demand' | 'planning' | 'mobile' | 'scenarios' | 'reports' | 'admin' | 'help'

const VIEW_IDS: ViewName[] = ['dashboard', 'map', 'demand', 'planning', 'mobile', 'scenarios', 'reports', 'admin', 'help']

function showView(view: ViewName) {
  const idx = VIEW_IDS.indexOf(view)
  if (idx < 0) return
  VIEW_IDS.forEach((v, i) => {
    const el = document.getElementById(`${v}-view`)
    if (el) (el as HTMLElement).hidden = i !== idx
  })
  VIEW_IDS.forEach((v, i) => {
    const el = document.getElementById(`nav-${v}`)
    if (el) el.classList.toggle('active', i === idx)
  })
  if (view === 'map') initMapIfNeeded()
  else if (view === 'demand') void initDemandIfNeeded()
  else if (view === 'planning') void initPlanningIfNeeded()
  else if (view === 'mobile') void initMobileIfNeeded()
  else if (view === 'scenarios') void initScenariosIfNeeded()
  else if (view === 'admin') void initAdminIfNeeded()
}

type AccessSummary = { metrics: AccessMetric[]; equity_gini: number; model: string }

async function loadMapWithAccessModel(model: string) {
  const summary = await fetchJson<AccessSummary>(`${API_BASE}/access/summary?model=${encodeURIComponent(model)}`)
  const facilities = await fetchJson<Facility[]>(`${API_BASE}/facilities`)
  const metrics = summary?.metrics ?? []
  const equityEl = document.getElementById('map-equity')
  if (equityEl) equityEl.textContent = `Equity (Gini): ${summary?.equity_gini?.toFixed(3) ?? '—'} (0 = equal, 1 = max inequality)`

  if (!mapInstance) return
  mapLayers.forEach((l) => l.remove())
  mapLayers = []
  if (facilities?.length) {
    facilities.forEach((f) => {
      const marker = window.L.marker([f.latitude, f.longitude])
        .addTo(mapInstance!)
        .bindPopup(`<strong>${f.name}</strong><br/>${f.facility_type} · ${f.bed_capacity} beds`)
      mapLayers.push(marker)
    })
  }
  const bounds: [number, number][] = []
  metrics.forEach((m) => {
    const color = accessScoreToColor(m.access_score)
    const circle = window.L.circleMarker([m.centroid_lat, m.centroid_lon], {
      radius: 10,
      fillColor: color,
      color: '#1e293b',
      weight: 1.5,
      fillOpacity: 0.85,
    })
      .addTo(mapInstance!)
      .bindPopup(
        `<strong>${m.region_name}</strong><br/>Population: ${m.population.toLocaleString()}<br/>` +
        `Distance: ${m.distance_km} km · Access score: ${m.access_score.toFixed(2)}<br/>Nearest: ${m.nearest_facility_name}`
      )
    mapLayers.push(circle)
    bounds.push([m.centroid_lat, m.centroid_lon])
  })
  if (bounds.length > 0) mapInstance.fitBounds(bounds as [number, number][], { padding: [30, 30] })
}

async function initMapIfNeeded() {
  const container = document.getElementById('map-container')
  if (!container || !window.L) return
  if (!mapInstance) {
    mapInstance = window.L.map('map-container').setView([20, 0], 2)
    window.L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
      attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>',
    }).addTo(mapInstance)
    const modelEl = document.getElementById('map-access-model') as HTMLSelectElement | null
    if (modelEl) {
      modelEl.addEventListener('change', () => void loadMapWithAccessModel(modelEl.value))
    }
  }
  mapInstance.invalidateSize()
  const modelEl = document.getElementById('map-access-model') as HTMLSelectElement | null
  await loadMapWithAccessModel(modelEl?.value ?? 'nearest')
}

type FacilityPlanResult = {
  num_new: number
  chosen_region_ids: string[]
  before_avg_distance_km: number
  after_avg_distance_km: number
  before_max_distance_km: number
  after_max_distance_km: number
  worst_region_improvement_km: number
}

function updatePlanningOverloadNote() {
  const noteEl = document.getElementById('planning-overload-note')
  if (!noteEl) return
  try {
    const raw = sessionStorage.getItem('overloadedRegionIds')
    const ids: string[] = raw ? JSON.parse(raw) : []
    if (ids.length === 0) {
      noteEl.hidden = true
      noteEl.innerHTML = ''
      return
    }
    noteEl.hidden = false
    noteEl.innerHTML = `<p class="overload-note-text">Suggested focus: regions from overloaded facilities (${ids.join(', ')}). <button type="button" id="planning-clear-overload" class="link-btn">Clear</button></p>`
    document.getElementById('planning-clear-overload')?.addEventListener('click', () => {
      sessionStorage.removeItem('overloadedRegionIds')
      updatePlanningOverloadNote()
    })
  } catch {
    noteEl.hidden = true
  }
}

async function initPlanningIfNeeded() {
  const selectEl = document.getElementById('planning-num-new') as HTMLSelectElement | null
  const runBtn = document.getElementById('planning-run-btn')
  if (!selectEl || !runBtn) return

  updatePlanningOverloadNote()

  const prioritizeEl = document.getElementById('planning-prioritize') as HTMLInputElement | null
  if (!runBtn.getAttribute('data-wired')) {
    runBtn.addEventListener('click', async () => {
      const numNew = Number(selectEl.value || '1')
      const prioritizeUnderserved = prioritizeEl?.checked ?? false
      await runFacilityPlanning(numNew, prioritizeUnderserved)
    })
    runBtn.setAttribute('data-wired', 'true')
  }
}

async function runFacilityPlanning(numNew: number, prioritizeUnderserved: boolean = false) {
  const resultBox = document.getElementById('planning-result')
  if (!resultBox) return
  resultBox.textContent = 'Running optimization...'

  const params = new URLSearchParams({ num_new: String(numNew) })
  if (prioritizeUnderserved) params.set('prioritize_underserved', 'true')
  const plan = await fetchJson<FacilityPlanResult>(
    `${API_BASE}/planning/facility?${params.toString()}`
  )
  const metrics = await fetchJson<AccessMetric[]>(`${API_BASE}/access/metrics`)
  const facilities = await fetchJson<Facility[]>(`${API_BASE}/facilities`)

  if (!plan || !metrics || !facilities) {
    resultBox.textContent = 'Could not run optimization. Check backend is running.'
    return
  }

  if (!plan.chosen_region_ids.length) {
    resultBox.textContent = 'No valid candidate regions found.'
    return
  }

  const chosen = metrics.filter((m) => plan.chosen_region_ids.includes(m.region_id))
  const listItems = chosen
    .map(
      (c) =>
        `<li><strong>${c.region_name}</strong> (ID ${c.region_id}) – centroid at ${c.centroid_lat.toFixed(
          3,
        )}, ${c.centroid_lon.toFixed(3)}</li>`
    )
    .join('')

  resultBox.innerHTML = `
    <p><strong>New facilities to add:</strong> ${plan.num_new}</p>
    <p><strong>Average distance:</strong> ${plan.before_avg_distance_km} km → ${plan.after_avg_distance_km} km</p>
    <p><strong>Worst-region distance:</strong> ${plan.before_max_distance_km} km → ${plan.after_max_distance_km} km</p>
    <p><strong>Worst-off region improvement:</strong> ${plan.worst_region_improvement_km} km</p>
    <p><strong>Suggested new facility sites (by region):</strong></p>
    <ul>${listItems}</ul>
    <button id="planning-save-scenario-btn" class="primary-btn">Save as scenario</button>
  `
  document.getElementById('planning-save-scenario-btn')?.addEventListener('click', async () => {
    const name = prompt('Scenario name')
    if (!name) return
    const body = {
      name,
      type: 'facility_plan',
      results_summary: plan,
      facility_results: chosen.map((c) => ({ region_id: c.region_id, region_name: c.region_name, centroid_lat: c.centroid_lat, centroid_lon: c.centroid_lon })),
    }
    const res = await fetchWithAuth(`${API_BASE}/scenarios`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) })
    if (res.ok) alert('Scenario saved.')
    else alert('Failed to save scenario.')
  })

  await renderPlanningMap(chosen, facilities)
}

async function renderPlanningMap(chosenRegions: AccessMetric[], facilities: Facility[]) {
  const container = document.getElementById('planning-map')
  if (!container || !window.L) return

  if (!planningMap) {
    planningMap = window.L.map('planning-map').setView([20, 0], 2)
    window.L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
      attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>',
    }).addTo(planningMap)
  }
  planningMap.invalidateSize()

  planningLayers.forEach((l) => l.remove())
  planningLayers = []

  facilities.forEach((f) => {
    const marker = window.L.marker([f.latitude, f.longitude], {
      title: f.name,
    }).bindPopup(`<strong>${f.name}</strong><br/>Existing facility`)
    marker.addTo(planningMap!)
    planningLayers.push(marker)
  })

  const bounds: [number, number][] = []
  chosenRegions.forEach((r) => {
    const marker = window.L.circleMarker([r.centroid_lat, r.centroid_lon], {
      radius: 10,
      fillColor: '#f97316',
      color: '#c2410c',
      weight: 2,
      fillOpacity: 0.9,
    }).bindPopup(
      `<strong>New facility candidate</strong><br/>Region: ${r.region_name}<br/>ID: ${r.region_id}`,
    )
    marker.addTo(planningMap!)
    planningLayers.push(marker)
    bounds.push([r.centroid_lat, r.centroid_lon])
  })

  if (bounds.length > 0) {
    planningMap.fitBounds(bounds as [number, number][], { padding: [30, 30] })
  }
}

type RouteStopDto = {
  region_id: string
  region_name: string
  latitude: number
  longitude: number
  sequence: number
}

type MobileRouteDto = {
  vehicle_id: number
  stops: RouteStopDto[]
  total_distance_km: number
}

type MobileRoutesResult = {
  depot_lat: number
  depot_lon: number
  routes: MobileRouteDto[]
}

const ROUTE_COLORS = ['#22c55e', '#3b82f6', '#a855f7', '#eab308']

async function initMobileIfNeeded() {
  const runBtn = document.getElementById('mobile-run-btn')
  const numVehiclesEl = document.getElementById('mobile-num-vehicles') as HTMLSelectElement | null
  if (!runBtn || !numVehiclesEl) return
  if (!runBtn.getAttribute('data-wired')) {
    runBtn.addEventListener('click', async () => {
      const numVehicles = Number(numVehiclesEl.value || '2')
      await runMobileRoutes(numVehicles)
    })
    runBtn.setAttribute('data-wired', 'true')
  }
}

async function runMobileRoutes(numVehicles: number) {
  const resultBox = document.getElementById('mobile-result')
  const tableBox = document.getElementById('mobile-routes-table')
  if (!resultBox || !tableBox) return
  resultBox.textContent = 'Computing routes...'
  tableBox.innerHTML = ''

  const data = await fetchJson<MobileRoutesResult>(
    `${API_BASE}/planning/mobile-routes?num_vehicles=${encodeURIComponent(numVehicles)}`
  )
  if (!data || !data.routes.length) {
    resultBox.textContent = 'Could not compute routes. Check backend.'
    return
  }

  resultBox.innerHTML = `Depot at (${data.depot_lat.toFixed(3)}, ${data.depot_lon.toFixed(3)}). ${data.routes.length} route(s). <button id="mobile-save-scenario-btn" class="primary-btn" style="margin-left:0.5rem">Save as scenario</button>`
  document.getElementById('mobile-save-scenario-btn')?.addEventListener('click', async () => {
    const name = prompt('Scenario name')
    if (!name) return
    const body = {
      name,
      type: 'mobile_routes',
      results_summary: { depot_lat: data.depot_lat, depot_lon: data.depot_lon, num_routes: data.routes.length },
      route_results: data.routes.map((r) => ({ vehicle_id: r.vehicle_id, stops: r.stops })),
    }
    const res = await fetchWithAuth(`${API_BASE}/scenarios`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) })
    if (res.ok) alert('Scenario saved.')
    else alert('Failed to save scenario.')
  })
  const rows = data.routes
    .map(
      (r) =>
        `<tr>
          <td>Vehicle ${r.vehicle_id + 1}</td>
          <td>${r.stops.map((s) => s.region_name).join(' → ')}</td>
          <td>${r.total_distance_km.toFixed(1)} km</td>
        </tr>`
    )
    .join('')
  tableBox.innerHTML = `
    <table>
      <thead><tr><th>Vehicle</th><th>Stops</th><th>Total distance</th></tr></thead>
      <tbody>${rows}</tbody>
    </table>
  `
  await renderMobileMap(data)
}

async function renderMobileMap(data: MobileRoutesResult) {
  const container = document.getElementById('mobile-map')
  if (!container || !window.L) return
  if (!mobileMap) {
    mobileMap = window.L.map('mobile-map').setView([data.depot_lat, data.depot_lon], 10)
    window.L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
      attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>',
    }).addTo(mobileMap)
  }
  mobileMap.invalidateSize()
  mobileLayers.forEach((l) => l.remove())
  mobileLayers = []

  const depotMarker = window.L.marker([data.depot_lat, data.depot_lon])
    .bindPopup('<strong>Depot</strong>')
    .addTo(mobileMap!)
  mobileLayers.push(depotMarker)

  const bounds: [number, number][] = [[data.depot_lat, data.depot_lon]]
  data.routes.forEach((route, idx) => {
    const color = ROUTE_COLORS[idx % ROUTE_COLORS.length]
    const points: [number, number][] = [[data.depot_lat, data.depot_lon]]
    route.stops.forEach((s) => {
      points.push([s.latitude, s.longitude])
      bounds.push([s.latitude, s.longitude])
      const m = window.L.circleMarker([s.latitude, s.longitude], {
        radius: 6,
        fillColor: color,
        color: '#0f172a',
        weight: 1,
        fillOpacity: 0.9,
      })
        .bindPopup(`<strong>${s.region_name}</strong> (stop ${s.sequence})`)
        .addTo(mobileMap!)
      mobileLayers.push(m)
    })
    points.push([data.depot_lat, data.depot_lon])
    const polyline = window.L.polyline(points, { color, weight: 4, opacity: 0.8 }).addTo(mobileMap!)
    mobileLayers.push(polyline)
  })
  if (bounds.length > 1) {
    mobileMap.fitBounds(bounds as [number, number][], { padding: [40, 40] })
  }
}

type DemandSummary = {
  facility_id: string
  facility_name: string
  facility_type: string
  region_id: string
  bed_capacity: number
  avg_monthly_visits: number
  forecast_next_month: number
  forecast_over_horizon: number
  utilization_ratio: number
  overload: boolean
  risk_score: number
  risk_level: string
}

function riskChip(level: string): string {
  const c = level === 'high' ? 'chip-risk-high' : level === 'medium' ? 'chip-risk-medium' : 'chip-risk-low'
  return `<span class="chip ${c}">${level}</span>`
}

async function fetchDemandSummary(timeHorizon: number, facilityType: string, overloadOnly: boolean): Promise<DemandSummary[]> {
  const params = new URLSearchParams()
  params.set('time_horizon_months', String(timeHorizon))
  if (facilityType) params.set('facility_type', facilityType)
  if (overloadOnly) params.set('overload_only', 'true')
  return (await fetchJson<DemandSummary[]>(`${API_BASE}/demand/summary?${params.toString()}`)) ?? []
}

function renderDemandTable(rows: DemandSummary[], tableContainer: HTMLElement, horizonMonths: number) {
  if (!rows.length) {
    tableContainer.innerHTML = 'No facilities match the filters.'
    return
  }
  const htmlRows = rows
    .map((r) => {
      const utilPct = (r.utilization_ratio * 100).toFixed(0)
      const flag = r.overload ? '<span class="chip chip-danger">Overload</span>' : '<span class="chip chip-ok">OK</span>'
      const risk = riskChip(r.risk_level)
      return `<tr data-facility-id="${r.facility_id}" data-facility-name="${r.facility_name}">
        <td>${r.facility_id}</td>
        <td>${r.facility_name}</td>
        <td>${r.facility_type}</td>
        <td>${r.bed_capacity}</td>
        <td>${r.forecast_over_horizon.toFixed(1)}</td>
        <td>${utilPct}%</td>
        <td>${risk}</td>
        <td>${flag}</td>
      </tr>`
    })
    .join('')
  tableContainer.innerHTML = `
    <table>
      <thead>
        <tr>
          <th>ID</th>
          <th>Facility</th>
          <th>Type</th>
          <th>Beds</th>
          <th>Forecast (${horizonMonths} mo)</th>
          <th>Utilization</th>
          <th>Risk</th>
          <th>Status</th>
        </tr>
      </thead>
      <tbody>${htmlRows}</tbody>
    </table>
    <p class="table-hint">Click a row to see the demand time series.</p>
  `
  tableContainer.querySelectorAll<HTMLTableRowElement>('tbody tr').forEach((row) => {
    row.addEventListener('click', () => {
      const id = row.dataset.facilityId
      const name = row.dataset.facilityName
      if (id && name) void loadDemandTimeseries(id, name)
    })
  })
}

async function initDemandIfNeeded() {
  const tableContainer = document.getElementById('demand-table-container')
  const timeHorizonEl = document.getElementById('demand-time-horizon') as HTMLSelectElement
  const facilityTypeEl = document.getElementById('demand-facility-type') as HTMLSelectElement
  const overloadOnlyEl = document.getElementById('demand-overload-only') as HTMLInputElement
  const sendToPlanningBtn = document.getElementById('demand-send-to-planning')
  if (!tableContainer) return

  async function applyDemandFilters() {
    const timeHorizon = Number(timeHorizonEl?.value || 1)
    const facilityType = (facilityTypeEl?.value || '').trim()
    const overloadOnly = overloadOnlyEl?.checked ?? false
    const rows = await fetchDemandSummary(timeHorizon, facilityType, overloadOnly)
    renderDemandTable(rows, tableContainer, timeHorizon)
  }

  if (!demandInitialized) {
    demandInitialized = true
    timeHorizonEl?.addEventListener('change', () => void applyDemandFilters())
    facilityTypeEl?.addEventListener('change', () => void applyDemandFilters())
    overloadOnlyEl?.addEventListener('change', () => void applyDemandFilters())
    sendToPlanningBtn?.addEventListener('click', async () => {
      const timeHorizon = Number(timeHorizonEl?.value || 1)
      const rows = await fetchDemandSummary(timeHorizon, '', true)
      const regionIds = [...new Set(rows.map((r) => r.region_id))]
      try {
        sessionStorage.setItem('overloadedRegionIds', JSON.stringify(regionIds))
      } catch (_) {}
      showView('planning')
    })
  }
  void applyDemandFilters()
}

type DemandPointDto = {
  date: string
  visits: number
}

async function loadDemandTimeseries(facilityId: string, facilityName: string) {
  const canvas = document.getElementById('demand-chart') as HTMLCanvasElement | null
  const titleEl = document.getElementById('demand-chart-title')
  if (!canvas || !window.Chart) return

  const points = await fetchJson<DemandPointDto[]>(`${API_BASE}/demand/timeseries/${facilityId}`)
  if (!points || points.length === 0) {
    if (titleEl) titleEl.textContent = `No historical demand data for ${facilityName}`
    if (demandChart) {
      demandChart.destroy()
      demandChart = null
    }
    return
  }

  const labels = points.map((p) => p.date)
  const data = points.map((p) => p.visits)

  if (titleEl) titleEl.textContent = `Historical demand for ${facilityName}`

  const ctx = canvas.getContext('2d')
  if (!ctx) return

  if (demandChart) {
    demandChart.destroy()
  }

  demandChart = new window.Chart(ctx, {
    type: 'line',
    data: {
      labels,
      datasets: [
        {
          label: 'Visits',
          data,
          borderColor: '#38bdf8',
          backgroundColor: 'rgba(56, 189, 248, 0.2)',
          tension: 0.2,
        },
      ],
    },
    options: {
      responsive: true,
      plugins: {
        legend: {
          display: false,
        },
      },
      scales: {
        x: {
          ticks: {
            color: '#9ca3af',
          },
        },
        y: {
          ticks: {
            color: '#9ca3af',
          },
        },
      },
    },
  })
}

type ScenarioListItem = { id: string; name: string; description: string | null; type: string; created_at: string; results_summary: Record<string, unknown> | null }
let scenariosListCache: ScenarioListItem[] = []

async function initScenariosIfNeeded() {
  const listEl = document.getElementById('scenarios-list')
  if (!listEl) return
  const list = await fetchJson<ScenarioListItem[]>(`${API_BASE}/scenarios`)
  scenariosListCache = list ?? []
  if (!list || list.length === 0) {
    listEl.innerHTML = '<p>No scenarios yet. Run Facility Planning or Mobile Routes and save a scenario from there.</p>'
    const selA = document.getElementById('compare-a') as HTMLSelectElement
    const selB = document.getElementById('compare-b') as HTMLSelectElement
    if (selA) selA.innerHTML = '<option value="">— Select —</option>'
    if (selB) selB.innerHTML = '<option value="">— Select —</option>'
    return
  }
  const rows = list.map((s) => `<tr><td>${s.name}</td><td>${s.type}</td><td>${new Date(s.created_at).toLocaleString()}</td><td><a href="#" data-id="${s.id}" class="scenario-open">Open</a></td></tr>`).join('')
  listEl.innerHTML = `<table><thead><tr><th>Name</th><th>Type</th><th>Created</th><th></th></tr></thead><tbody>${rows}</tbody></table>`
  listEl.querySelectorAll('.scenario-open').forEach((a) => {
    a.addEventListener('click', (e) => {
      e.preventDefault()
      const id = (a as HTMLElement).dataset.id
      if (id) window.open(`${API_BASE}/scenarios/${id}`, '_blank')
    })
  })
  const selA = document.getElementById('compare-a') as HTMLSelectElement
  const selB = document.getElementById('compare-b') as HTMLSelectElement
  const opts = list.map((s) => `<option value="${s.id}">${s.name} (${s.type})</option>`).join('')
  if (selA) selA.innerHTML = '<option value="">— Select —</option>' + opts
  if (selB) selB.innerHTML = '<option value="">— Select —</option>' + opts
  const compareBtn = document.getElementById('compare-btn')
  const compareResult = document.getElementById('scenarios-compare-result')
  if (compareBtn && compareResult) {
    compareBtn.addEventListener('click', async () => {
      const idA = selA?.value
      const idB = selB?.value
      if (!idA || !idB || idA === idB) {
        compareResult.innerHTML = '<p>Select two different scenarios.</p>'
        return
      }
      const [detailA, detailB] = await Promise.all([
        fetchJson<{ name: string; type: string; results_summary: Record<string, unknown> | null }>(`${API_BASE}/scenarios/${idA}`),
        fetchJson<{ name: string; type: string; results_summary: Record<string, unknown> | null }>(`${API_BASE}/scenarios/${idB}`),
      ])
      if (!detailA || !detailB) {
        compareResult.innerHTML = '<p>Could not load one or both scenarios.</p>'
        return
      }
      const fmt = (o: Record<string, unknown> | null) => {
        if (!o) return '—'
        const lines = Object.entries(o).map(([k, v]) => `${k}: ${typeof v === 'number' ? (v as number).toFixed(2) : String(v)}`)
        return lines.join('<br/>')
      }
      compareResult.innerHTML = `
        <div class="compare-grid">
          <div class="panel"><h3>${detailA.name}</h3><p class="compare-type">${detailA.type}</p><div class="compare-metrics">${fmt(detailA.results_summary)}</div></div>
          <div class="panel"><h3>${detailB.name}</h3><p class="compare-type">${detailB.type}</p><div class="compare-metrics">${fmt(detailB.results_summary)}</div></div>
        </div>
      `
    })
  }
}

async function initAdminIfNeeded() {
  const auditEl = document.getElementById('admin-audit')
  const usersEl = document.getElementById('admin-users')
  if (auditEl) {
    const logs = await fetchJson<{ id: string; action_type: string; details: unknown; created_at: string }[]>(`${API_BASE}/admin/audit-logs`)
    if (logs?.length) {
      auditEl.innerHTML = `<table><thead><tr><th>Time</th><th>Action</th><th>Details</th></tr></thead><tbody>${
        logs.slice(0, 50).map((l) => `<tr><td>${new Date(l.created_at).toLocaleString()}</td><td>${l.action_type}</td><td>${JSON.stringify(l.details ?? {})}</td></tr>`).join('')
      }</tbody></table>`
    } else {
      auditEl.textContent = 'No audit entries or backend error.'
    }
  }
  if (usersEl) {
    const users = await fetchJson<{ id: string; email: string | null; role: string }[]>(`${API_BASE}/admin/users`)
    if (users?.length) {
      usersEl.innerHTML = `<table><thead><tr><th>Email</th><th>Role</th></tr></thead><tbody>${users.map((u) => `<tr><td>${u.email ?? u.id}</td><td>${u.role}</td></tr>`).join('')}</tbody></table>`
    } else {
      usersEl.textContent = 'No users (add via Supabase Auth and profiles).'
    }
  }
  const rBtn = document.getElementById('admin-upload-regions-btn')
  const rInput = document.getElementById('admin-upload-regions') as HTMLInputElement
  if (rBtn && rInput) {
    rBtn.addEventListener('click', async () => {
      if (!rInput.files?.[0]) return
      const form = new FormData()
      form.append('file', rInput.files[0])
      const res = await fetchWithAuth(`${API_BASE}/admin/upload/regions`, { method: 'POST', body: form })
      const data = await res.json().catch(() => ({}))
      alert(data.uploaded != null ? `Uploaded ${data.uploaded} rows` : 'Upload failed')
    })
  }
  const fBtn = document.getElementById('admin-upload-facilities-btn')
  const fInput = document.getElementById('admin-upload-facilities') as HTMLInputElement
  if (fBtn && fInput) {
    fBtn.addEventListener('click', async () => {
      if (!fInput.files?.[0]) return
      const form = new FormData()
      form.append('file', fInput.files[0])
      const res = await fetchWithAuth(`${API_BASE}/admin/upload/facilities`, { method: 'POST', body: form })
      const data = await res.json().catch(() => ({}))
      alert(data.uploaded != null ? `Uploaded ${data.uploaded} rows` : 'Upload failed')
    })
  }
  const dBtn = document.getElementById('admin-upload-demand-btn')
  const dInput = document.getElementById('admin-upload-demand') as HTMLInputElement
  if (dBtn && dInput) {
    dBtn.addEventListener('click', async () => {
      if (!dInput.files?.[0]) return
      const form = new FormData()
      form.append('file', dInput.files[0])
      const res = await fetchWithAuth(`${API_BASE}/admin/upload/demand`, { method: 'POST', body: form })
      const data = await res.json().catch(() => ({}))
      alert(data.uploaded != null ? `Uploaded ${data.uploaded} rows` : 'Upload failed')
    })
  }
}

type DashboardSummary = {
  total_population: number
  population_adequate_access: number
  pct_adequate: number
  underserved_region_count: number
  overloaded_facility_count: number
  alerts: string[]
  last_updated_regions: string | null
  last_updated_facilities: string | null
  last_updated_demand: string | null
}

async function renderDashboard(
  statusEl: HTMLDivElement,
  regionsEl: HTMLDivElement,
  facilitiesEl: HTMLDivElement
) {
  const health = await fetchJson<{ status: string; message: string }>(`${API_BASE}/health`)
  if (health) {
    statusEl.textContent = `Backend: ${health.status} – ${health.message}`
  } else {
    statusEl.textContent = `Could not reach backend at ${API_BASE}/health. Is it running?`
  }

  const summary = await fetchJson<DashboardSummary>(`${API_BASE}/dashboard/summary`)
  const kpisEl = document.getElementById('dashboard-kpis')
  const alertsEl = document.getElementById('dashboard-alerts-list')
  const lastUpdatedEl = document.getElementById('dashboard-last-updated')
  if (summary && kpisEl) {
    kpisEl.innerHTML = `
      <div class="kpi-card"><span class="kpi-value">${summary.total_population.toLocaleString()}</span><span class="kpi-label">Total population</span></div>
      <div class="kpi-card"><span class="kpi-value">${summary.pct_adequate}%</span><span class="kpi-label">Population with adequate access</span></div>
      <div class="kpi-card"><span class="kpi-value">${summary.underserved_region_count}</span><span class="kpi-label">Underserved regions</span></div>
      <div class="kpi-card"><span class="kpi-value">${summary.overloaded_facility_count}</span><span class="kpi-label">Facilities predicted overloaded</span></div>
    `
  }
  if (alertsEl) {
    if (summary?.alerts?.length) {
      alertsEl.innerHTML = `<ul>${summary.alerts.map((a) => `<li>${a}</li>`).join('')}</ul>`
    } else {
      alertsEl.textContent = summary ? 'No alerts.' : 'Could not load alerts.'
    }
  }
  if (lastUpdatedEl) {
    const parts = []
    if (summary?.last_updated_regions) parts.push(`Regions: ${summary.last_updated_regions}`)
    if (summary?.last_updated_facilities) parts.push(`Facilities: ${summary.last_updated_facilities}`)
    if (summary?.last_updated_demand) parts.push(`Demand: ${summary.last_updated_demand}`)
    lastUpdatedEl.textContent = parts.length ? `Last data updated: ${parts.join(' · ')}` : 'Last data updated: —'
  }

  const regions = await fetchJson<Region[]>(`${API_BASE}/regions`)
  if (!regions || regions.length === 0) {
    regionsEl.textContent = 'No regions found.'
  } else {
    const rows = regions
      .map(
        (r) =>
          `<tr><td>${r.id}</td><td>${r.name}</td><td>${r.population.toLocaleString()}</td></tr>`
      )
      .join('')
    regionsEl.innerHTML = `
      <table>
        <thead>
          <tr><th>ID</th><th>Name</th><th>Population</th></tr>
        </thead>
        <tbody>${rows}</tbody>
      </table>
    `
  }

  const facilities = await fetchJson<Facility[]>(`${API_BASE}/facilities`)
  if (!facilities || facilities.length === 0) {
    facilitiesEl.textContent = 'No facilities found.'
  } else {
    const rows = facilities
      .map(
        (f) =>
          `<tr><td>${f.id}</td><td>${f.name}</td><td>${f.facility_type}</td><td>${f.bed_capacity}</td></tr>`
      )
      .join('')
    facilitiesEl.innerHTML = `
      <table>
        <thead>
          <tr><th>ID</th><th>Name</th><th>Type</th><th>Beds</th></tr>
        </thead>
        <tbody>${rows}</tbody>
      </table>
    `
  }
}

function renderLoginScreen() {
  if (!app) return
  app.innerHTML = `
    <main class="app-root auth-screen">
      <div class="auth-card">
        <h1>Sign in</h1>
        <p class="auth-subtitle">Healthcare Access Inequality Analysis & Planning System</p>
        <form id="login-form" class="auth-form">
          <label for="login-email">Email</label>
          <input type="email" id="login-email" required placeholder="you@example.com" />
          <label for="login-password">Password</label>
          <input type="password" id="login-password" required />
          <p id="login-error" class="auth-error" hidden></p>
          <button type="submit" class="primary-btn">Sign in</button>
        </form>
        <p class="auth-switch">Don't have an account? <a href="#" id="auth-go-signup">Sign up</a></p>
      </div>
    </main>
  `
  const form = document.getElementById('login-form')
  const emailEl = document.getElementById('login-email') as HTMLInputElement
  const passwordEl = document.getElementById('login-password') as HTMLInputElement
  const errEl = document.getElementById('login-error')
  form?.addEventListener('submit', async (e) => {
    e.preventDefault()
    if (errEl) { errEl.hidden = true; errEl.textContent = '' }
    const { error } = await supabase.auth.signInWithPassword({ email: emailEl.value.trim(), password: passwordEl.value })
    if (error) {
      if (errEl) { errEl.hidden = false; errEl.textContent = error.message }
      return
    }
    void bootstrap()
  })
  document.getElementById('auth-go-signup')?.addEventListener('click', (e) => {
    e.preventDefault()
    renderSignupScreen()
  })
}

function renderSignupScreen() {
  if (!app) return
  app.innerHTML = `
    <main class="app-root auth-screen">
      <div class="auth-card">
        <h1>Sign up</h1>
        <p class="auth-subtitle">Create an account (email + password)</p>
        <form id="signup-form" class="auth-form">
          <label for="signup-email">Email</label>
          <input type="email" id="signup-email" required placeholder="you@example.com" />
          <label for="signup-password">Password</label>
          <input type="password" id="signup-password" required minlength="6" />
          <p id="signup-error" class="auth-error" hidden></p>
          <button type="submit" class="primary-btn">Sign up</button>
        </form>
        <p class="auth-switch">Already have an account? <a href="#" id="auth-go-login">Sign in</a></p>
      </div>
    </main>
  `
  const form = document.getElementById('signup-form')
  const emailEl = document.getElementById('signup-email') as HTMLInputElement
  const passwordEl = document.getElementById('signup-password') as HTMLInputElement
  const errEl = document.getElementById('signup-error')
  form?.addEventListener('submit', async (e) => {
    e.preventDefault()
    if (errEl) { errEl.hidden = true; errEl.textContent = '' }
    const { error } = await supabase.auth.signUp({ email: emailEl.value.trim(), password: passwordEl.value })
    if (error) {
      if (errEl) { errEl.hidden = false; errEl.textContent = error.message }
      return
    }
    void bootstrap()
  })
  document.getElementById('auth-go-login')?.addEventListener('click', (e) => {
    e.preventDefault()
    renderLoginScreen()
  })
}

async function bootstrap() {
  if (!app) return
  const { data: { session } } = await supabase.auth.getSession()
  if (!session) {
    renderLoginScreen()
    return
  }
  const token = session.access_token
  const res = await fetchWithAuth(`${API_BASE}/auth/me`)
  if (!res.ok) {
    currentUser = { user_id: session.user.id, email: session.user.email ?? '', role: 'viewer' }
  } else {
    const me = await res.json() as AuthUser
    currentUser = me
  }
  init()
}

async function init() {
  if (!app) return

  app.innerHTML = `
    <main class="app-root">
      <nav class="app-nav" aria-label="Main">
        <a href="#" id="nav-dashboard" class="nav-link active"><span class="nav-icon">🏠</span><span>Dashboard</span></a>
        <a href="#" id="nav-map" class="nav-link"><span class="nav-icon">🗺️</span><span>Access & Inequality</span></a>
        <a href="#" id="nav-demand" class="nav-link"><span class="nav-icon">📈</span><span>Demand & Overload</span></a>
        <a href="#" id="nav-planning" class="nav-link"><span class="nav-icon">🏥</span><span>Facility Planning</span></a>
        <a href="#" id="nav-mobile" class="nav-link"><span class="nav-icon">🚑</span><span>Mobile Units</span></a>
        <a href="#" id="nav-scenarios" class="nav-link"><span class="nav-icon">💾</span><span>Scenarios</span></a>
        <a href="#" id="nav-reports" class="nav-link"><span class="nav-icon">📄</span><span>Reports</span></a>
        <a href="#" id="nav-admin" class="nav-link"><span class="nav-icon">⚙️</span><span>Admin</span></a>
        <a href="#" id="nav-help" class="nav-link"><span class="nav-icon">❓</span><span>Help</span></a>
        <span class="nav-user">${currentUser?.email ?? ''} <button type="button" id="nav-logout" class="link-btn">Logout</button></span>
      </nav>

      <div id="dashboard-view">
        <h1>Healthcare Access Inequality Analysis & Planning System</h1>
        <p>Overview and key metrics.</p>
        <div id="backend-status" class="status-box">Checking backend status...</div>
        <section id="dashboard-kpis" class="dashboard-kpis"></section>
        <section id="dashboard-alerts" class="panel dashboard-alerts" aria-label="Alerts">
          <h2>Alerts</h2>
          <div id="dashboard-alerts-list">Loading…</div>
        </section>
        <p id="dashboard-last-updated" class="dashboard-meta"></p>
        <p class="export-links">
          <a href="${API_BASE}/export/regions" download="regions.csv" class="export-link">Export regions CSV</a>
          <a href="${API_BASE}/export/facilities" download="facilities.csv" class="export-link">Export facilities CSV</a>
        </p>
        <section class="layout-grid" aria-label="Demo data">
          <section class="panel" aria-label="Regions">
            <h2>Regions</h2>
            <div id="regions-container">Loading regions...</div>
          </section>
          <section class="panel" aria-label="Facilities">
            <h2>Facilities</h2>
            <div id="facilities-container">Loading facilities...</div>
          </section>
        </section>
      </div>

      <div id="map-view" hidden>
        <h1>Access & Inequality Map</h1>
        <p class="map-legend">Regions colored by access score (green = better, red = worse). Markers are facilities.</p>
        <p class="map-controls">
          <label for="map-access-model">Access model:</label>
          <select id="map-access-model">
            <option value="nearest">Nearest distance</option>
            <option value="2sfca">2SFCA</option>
          </select>
          <span id="map-equity" class="equity-display"></span>
        </p>
        <p><a href="${API_BASE}/export/access" download="access_metrics.csv" class="export-link">Export access CSV</a></p>
        <div id="map-container"></div>
      </div>

      <div id="demand-view" hidden>
        <h1>Demand & Overload</h1>
        <p class="map-legend">Forecasts and risk by facility. Use filters to focus on time horizon, type, or overloaded only.</p>
        <p class="map-controls">
          <label for="demand-time-horizon">Time horizon:</label>
          <select id="demand-time-horizon">
            <option value="1">1 month</option>
            <option value="3">3 months</option>
            <option value="6" selected>6 months</option>
            <option value="12">12 months</option>
          </select>
          <label for="demand-facility-type">Facility type:</label>
          <select id="demand-facility-type">
            <option value="">All</option>
            <option value="hospital">Hospital</option>
            <option value="clinic">Clinic</option>
          </select>
          <label><input type="checkbox" id="demand-overload-only" /> Overloaded only</label>
          <button id="demand-send-to-planning" class="primary-btn">Use overloaded in Facility Planning</button>
        </p>
        <p><a href="${API_BASE}/export/demand" download="demand_summary.csv" class="export-link">Export demand CSV</a></p>
        <section class="layout-grid demand-layout" aria-label="Demand and overload">
          <section class="panel" aria-label="Demand summary">
            <h2>Facility demand summary</h2>
            <div id="demand-table-container">Loading demand data...</div>
          </section>
          <section class="panel" aria-label="Time series chart">
            <h2 id="demand-chart-title">Select a facility to see its history</h2>
            <canvas id="demand-chart"></canvas>
          </section>
        </section>
      </div>

      <div id="planning-view" hidden>
        <h1>Facility Planning</h1>
        <p class="map-legend">
          Simple optimization that suggests new facility sites (by region) to reduce population-weighted distance.
        </p>
        <section class="layout-grid planning-layout" aria-label="Facility planning">
          <section class="panel" aria-label="Planning controls and results">
            <h2>Run facility placement optimization</h2>
            <div id="planning-overload-note" class="planning-overload-note" hidden></div>
            <label for="planning-num-new">Number of new facilities to consider:</label>
            <select id="planning-num-new">
              <option value="1">1</option>
              <option value="2">2</option>
              <option value="3">3</option>
            </select>
            <p class="checkbox-row">
              <input type="checkbox" id="planning-prioritize" />
              <label for="planning-prioritize">Prioritize underserved regions (equity)</label>
            </p>
            <button id="planning-run-btn" class="primary-btn">Run optimization</button>
            <div id="planning-result" class="planning-result-box">
              Choose how many new facilities to add and click “Run optimization”.
            </div>
          </section>
          <section class="panel" aria-label="Planning map">
            <h2>Map: existing vs proposed sites</h2>
            <div id="planning-map"></div>
          </section>
        </section>
      </div>

      <div id="mobile-view" hidden>
        <h1>Mobile Units & Routes</h1>
        <p class="map-legend">
          Plan routes for mobile health units visiting region centroids. Depot is the first facility.
        </p>
        <section class="layout-grid mobile-layout" aria-label="Mobile routing">
          <section class="panel" aria-label="Routing controls">
            <h2>Run routing</h2>
            <label for="mobile-num-vehicles">Number of vehicles:</label>
            <select id="mobile-num-vehicles">
              <option value="1">1</option>
              <option value="2" selected>2</option>
              <option value="3">3</option>
            </select>
            <button id="mobile-run-btn" class="primary-btn">Run routing</button>
            <div id="mobile-result" class="planning-result-box">Choose number of vehicles and click Run routing.</div>
            <div id="mobile-routes-table"></div>
          </section>
          <section class="panel" aria-label="Routes map">
            <h2>Map: depot and routes</h2>
            <div id="mobile-map"></div>
          </section>
        </section>
    </div>

      <div id="scenarios-view" hidden>
        <h1>Scenarios</h1>
        <p class="map-legend">Saved facility plans and mobile route plans.</p>
        <div id="scenarios-list">Loading…</div>
        <section class="panel scenarios-compare" aria-label="Compare scenarios">
          <h2>Compare two scenarios</h2>
          <p class="map-legend">Select two scenarios to compare key metrics side-by-side.</p>
          <p class="map-controls">
            <label for="compare-a">Scenario A</label>
            <select id="compare-a"><option value="">— Select —</option></select>
            <label for="compare-b">Scenario B</label>
            <select id="compare-b"><option value="">— Select —</option></select>
            <button id="compare-btn" class="primary-btn">Compare</button>
          </p>
          <div id="scenarios-compare-result"></div>
        </section>
      </div>

      <div id="reports-view" hidden>
        <h1>Reports & Exports</h1>
        <p class="map-legend">Download data as CSV.</p>
        <ul class="export-list">
          <li><a href="${API_BASE}/export/regions" download="regions.csv" class="export-link">Regions</a></li>
          <li><a href="${API_BASE}/export/facilities" download="facilities.csv" class="export-link">Facilities</a></li>
          <li><a href="${API_BASE}/export/access" download="access_metrics.csv" class="export-link">Access metrics</a></li>
          <li><a href="${API_BASE}/export/demand" download="demand_summary.csv" class="export-link">Demand summary</a></li>
        </ul>
      </div>

      <div id="admin-view" hidden>
        <h1>Admin</h1>
        <p class="map-legend">Audit log, config, data upload, users.</p>
        <div class="admin-tabs">
          <section class="panel">
            <h2>Audit log</h2>
            <div id="admin-audit">Loading…</div>
          </section>
          <section class="panel">
            <h2>Upload data (CSV)</h2>
            <p>Regions: id, name, country_code, population</p>
            <input type="file" id="admin-upload-regions" accept=".csv" />
            <button id="admin-upload-regions-btn" class="primary-btn">Upload regions</button>
            <p>Facilities: id, name, facility_type, region_id, latitude, longitude, bed_capacity</p>
            <input type="file" id="admin-upload-facilities" accept=".csv" />
            <button id="admin-upload-facilities-btn" class="primary-btn">Upload facilities</button>
            <p>Demand: date, facility_id, visits</p>
            <input type="file" id="admin-upload-demand" accept=".csv" />
            <button id="admin-upload-demand-btn" class="primary-btn">Upload demand</button>
          </section>
          <section class="panel">
            <h2>Users</h2>
            <div id="admin-users">Loading…</div>
          </section>
        </div>
      </div>

      <div id="help-view" hidden>
        <h1>Help / About</h1>
        <p class="map-legend">Healthcare Access Inequality Analysis & Planning System.</p>
        <p><strong>Access & Inequality Map:</strong> Choose nearest-distance or 2SFCA model; regions are colored by access score. Gini shows equity (0 = equal).</p>
        <p><strong>Demand & Overload:</strong> Forecasts use average historical visits; facilities over capacity are flagged.</p>
        <p><strong>Facility Planning:</strong> Suggests new facility sites to reduce population-weighted distance; optionally prioritize underserved regions.</p>
        <p><strong>Mobile Units & Routes:</strong> VRP routes from depot (first facility) to region centroids.</p>
        <p><strong>Scenarios:</strong> Save facility or route plans for comparison.</p>
  </div>
    </main>
  `

  const statusEl = document.querySelector<HTMLDivElement>('#backend-status')!
  const regionsEl = document.querySelector<HTMLDivElement>('#regions-container')!
  const facilitiesEl = document.querySelector<HTMLDivElement>('#facilities-container')!

  await renderDashboard(statusEl, regionsEl, facilitiesEl)

  document.getElementById('nav-dashboard')?.addEventListener('click', (e) => {
    e.preventDefault()
    showView('dashboard')
  })
  document.getElementById('nav-map')?.addEventListener('click', (e) => {
    e.preventDefault()
    showView('map')
  })
  document.getElementById('nav-demand')?.addEventListener('click', (e) => {
    e.preventDefault()
    showView('demand')
  })
  document.getElementById('nav-planning')?.addEventListener('click', (e) => {
    e.preventDefault()
    showView('planning')
  })
  document.getElementById('nav-mobile')?.addEventListener('click', (e) => {
    e.preventDefault()
    showView('mobile')
  })
  document.getElementById('nav-scenarios')?.addEventListener('click', (e) => {
    e.preventDefault()
    showView('scenarios')
  })
  document.getElementById('nav-reports')?.addEventListener('click', (e) => {
    e.preventDefault()
    showView('reports')
  })
  document.getElementById('nav-admin')?.addEventListener('click', (e) => {
    e.preventDefault()
    showView('admin')
  })
  document.getElementById('nav-help')?.addEventListener('click', (e) => {
    e.preventDefault()
    showView('help')
  })
  document.getElementById('nav-logout')?.addEventListener('click', async () => {
    await supabase.auth.signOut()
    currentUser = null
    renderLoginScreen()
  })
}

void bootstrap()
