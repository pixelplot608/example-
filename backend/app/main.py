import csv
import io
import math
from datetime import date, datetime
from uuid import UUID

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel
from typing import List, Optional, Any

from .db import get_pool, close_pool


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Return distance in km between two (lat, lon) points."""
    R = 6371.0  # Earth radius in km
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


OVERLOAD_UTILIZATION_THRESHOLD = 1.0  # > 1.0 == forecast demand exceeds capacity


app = FastAPI(title="Healthcare Access Inequality API")

# For now, allow all origins while developing locally.
# Later, we can restrict this to the frontend domain.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health_check():
    """
    Simple health/status endpoint so the frontend
    can verify that the backend is running.
    """
    return {"status": "ok", "message": "Healthcare Access API is running"}


@app.get("/")
async def root():
    return {"message": "Healthcare Access Inequality Analysis & Planning System API"}


# --- Dashboard summary (KPIs + alerts) ---
ADEQUATE_ACCESS_SCORE_THRESHOLD = 0.15  # regions with score >= this count as "adequate"
LOW_ACCESS_ALERT_THRESHOLD = 0.10  # regions below this generate an alert


class DashboardSummary(BaseModel):
    total_population: int
    population_adequate_access: int
    pct_adequate: float
    underserved_region_count: int
    overloaded_facility_count: int
    alerts: List[str]
    last_updated_regions: Optional[str] = None
    last_updated_facilities: Optional[str] = None
    last_updated_demand: Optional[str] = None


@app.get("/dashboard/summary", response_model=DashboardSummary)
async def dashboard_summary() -> DashboardSummary:
    """
    KPIs and alerts for the Dashboard: total pop, % adequate access,
    underserved count, overloaded count, and alert messages.
    """
    pool = await get_pool()
    # Total population from regions
    pop_row = await pool.fetchrow("SELECT COALESCE(SUM(population), 0)::bigint AS total FROM regions")
    total_pop = int(pop_row["total"]) if pop_row else 0

    # Access metrics for adequate/underserved
    metrics = await get_access_metrics()
    population_adequate = sum(m.population for m in metrics if m.access_score >= ADEQUATE_ACCESS_SCORE_THRESHOLD)
    underserved_count = sum(1 for m in metrics if m.access_score < ADEQUATE_ACCESS_SCORE_THRESHOLD)
    pct = (100.0 * population_adequate / total_pop) if total_pop > 0 else 0.0

    # Overload from demand summary
    summaries = await demand_summary()
    overloaded = [s for s in summaries if s.overload]
    overloaded_count = len(overloaded)

    # Alerts
    alerts: List[str] = []
    for s in overloaded:
        alerts.append(f"{s.facility_name} predicted to exceed capacity (utilization {s.utilization_ratio:.0%})")
    for m in metrics:
        if m.access_score < LOW_ACCESS_ALERT_THRESHOLD:
            alerts.append(f"{m.region_name} has very low access score ({m.access_score:.2f})")

    return DashboardSummary(
        total_population=total_pop,
        population_adequate_access=population_adequate,
        pct_adequate=round(pct, 1),
        underserved_region_count=underserved_count,
        overloaded_facility_count=overloaded_count,
        alerts=alerts[:20],
        last_updated_regions=None,
        last_updated_facilities=None,
        last_updated_demand=None,
    )


@app.on_event("startup")
async def on_startup() -> None:
    # Initialise the database connection pool
    await get_pool()


@app.on_event("shutdown")
async def on_shutdown() -> None:
    # Close the database connection pool
    await close_pool()


class Region(BaseModel):
    id: str
    name: str
    country_code: str
    population: int


class Facility(BaseModel):
    id: str
    name: str
    facility_type: str  # e.g. "hospital", "clinic"
    region_id: str
    latitude: float
    longitude: float
    bed_capacity: int


class AccessMetric(BaseModel):
    """Access metric for one region: distance to nearest facility and derived score."""
    region_id: str
    region_name: str
    population: int
    centroid_lat: float
    centroid_lon: float
    distance_km: float
    access_score: float  # 0–1, higher = better access
    nearest_facility_id: str
    nearest_facility_name: str


class AccessSummaryResponse(BaseModel):
    """Access metrics plus equity (Gini) and model name."""
    metrics: List[AccessMetric]
    equity_gini: float  # 0 = equal access, 1 = max inequality
    model: str  # "nearest" | "2sfca"


class DemandPoint(BaseModel):
    """Historical demand for one facility on a given date."""

    date: date
    visits: int


class DemandForecastSummary(BaseModel):
    """Simple forecast / overload summary for a facility."""

    facility_id: str
    facility_name: str
    facility_type: str
    region_id: str
    bed_capacity: int
    avg_monthly_visits: float
    forecast_next_month: float
    forecast_over_horizon: float  # forecast for next N months (avg_monthly * N)
    utilization_ratio: float
    overload: bool
    risk_score: float  # 0-1+ (same as utilization for now)
    risk_level: str  # "low" | "medium" | "high"


class FacilityPlanResult(BaseModel):
    """Simple facility planning result summarising new sites and distance impact."""

    num_new: int
    chosen_region_ids: List[str]
    before_avg_distance_km: float
    after_avg_distance_km: float
    before_max_distance_km: float
    after_max_distance_km: float
    worst_region_improvement_km: float  # before_max - after_max (equity improvement)


# --- Audit (Phase 8) ---


async def audit_log(action_type: str, details: Optional[dict] = None, user_id: Optional[UUID] = None) -> None:
    try:
        pool = await get_pool()
        await pool.execute(
            "INSERT INTO audit_logs (user_id, action_type, details) VALUES ($1, $2, $3)",
            user_id,
            action_type,
            details or {},
        )
    except Exception:
        pass  # do not fail requests if audit fails (e.g. table missing)


# --- Scenarios (Phase 7) ---


class ScenarioFacilityResult(BaseModel):
    region_id: str
    region_name: str
    centroid_lat: float
    centroid_lon: float


class ScenarioRouteStop(BaseModel):
    region_id: str
    region_name: str
    latitude: float
    longitude: float
    sequence: int
    leg_distance_km: Optional[float] = None


class ScenarioCreate(BaseModel):
    name: str
    description: Optional[str] = None
    type: str  # facility_plan | mobile_routes | combined
    inputs: Optional[dict] = None
    results_summary: Optional[dict] = None
    facility_results: Optional[List[ScenarioFacilityResult]] = None
    route_results: Optional[List[dict]] = None  # list of { vehicle_id, stops: [ScenarioRouteStop] }


class ScenarioListItem(BaseModel):
    id: UUID
    name: str
    description: Optional[str]
    type: str
    created_at: datetime
    results_summary: Optional[dict]


class ScenarioDetail(BaseModel):
    id: UUID
    name: str
    description: Optional[str]
    type: str
    created_by: Optional[UUID]
    created_at: datetime
    inputs: Optional[dict]
    results_summary: Optional[dict]
    facility_results: List[ScenarioFacilityResult]
    route_results: List[dict]


@app.get("/regions", response_model=List[Region])
async def list_regions() -> List[Region]:
    """
    Return a list of regions from the database.
    """
    pool = await get_pool()
    rows = await pool.fetch(
        """
        SELECT id, name, country_code, population
        FROM regions
        ORDER BY id
        """
    )
    return [Region(**dict(row)) for row in rows]


@app.get("/facilities", response_model=List[Facility])
async def list_facilities() -> List[Facility]:
    """
    Return a list of health facilities from the database.
    """
    pool = await get_pool()
    rows = await pool.fetch(
        """
        SELECT
            id,
            name,
            facility_type,
            region_id,
            latitude,
            longitude,
            bed_capacity
        FROM facilities
        ORDER BY id
        """
    )
    return [Facility(**dict(row)) for row in rows]


def _gini_coefficient(metrics: List[AccessMetric]) -> float:
    """Gini coefficient of access scores (population-weighted). 0 = equality, 1 = max inequality."""
    if not metrics:
        return 0.0
    total_pop = sum(m.population for m in metrics)
    if total_pop == 0:
        return 0.0
    # Sort by access_score ascending
    sorted_m = sorted(metrics, key=lambda m: m.access_score)
    cum_pop = 0.0
    cum_score = 0.0
    total_score = sum(m.population * m.access_score for m in metrics)
    if total_score == 0:
        return 0.0
    gini = 0.0
    for m in sorted_m:
        cum_pop += m.population
        cum_score += m.population * m.access_score
        gini += (2 * cum_pop - m.population) * m.population * m.access_score
    gini = 1.0 - gini / (total_pop * total_score)
    return round(max(0.0, min(1.0, gini)), 4)


def _compute_nearest_metrics(
    region_list: List[dict],
    facility_list: List[dict],
    region_facilities: dict[str, list],
) -> List[AccessMetric]:
    result: List[AccessMetric] = []
    for r in region_list:
        rid = r["id"]
        facs_in_region = region_facilities.get(rid, [])
        if not facs_in_region:
            continue
        clat = sum(f["latitude"] for f in facs_in_region) / len(facs_in_region)
        clon = sum(f["longitude"] for f in facs_in_region) / len(facs_in_region)
        best_km = float("inf")
        best_fac: Optional[dict] = None
        for f in facility_list:
            d = haversine_km(clat, clon, f["latitude"], f["longitude"])
            if d < best_km:
                best_km = d
                best_fac = f
        if best_fac is None:
            continue
        access_score = 1.0 / (1.0 + best_km)
        result.append(
            AccessMetric(
                region_id=rid,
                region_name=r["name"],
                population=r["population"],
                centroid_lat=round(clat, 6),
                centroid_lon=round(clon, 6),
                distance_km=round(best_km, 2),
                access_score=round(access_score, 4),
                nearest_facility_id=best_fac["id"],
                nearest_facility_name=best_fac["name"],
            )
        )
    return result


def _compute_2sfca_metrics(
    region_list: List[dict],
    facility_list: List[dict],
    region_facilities: dict[str, list],
) -> List[AccessMetric]:
    """2SFCA: supply-to-demand ratio with decay 1/(1+d_km)."""
    decay = lambda d: 1.0 / (1.0 + d) if d >= 0 else 0.0
    centroids: dict[str, tuple[float, float]] = {}
    for r in region_list:
        rid = r["id"]
        facs = region_facilities.get(rid, [])
        if not facs:
            continue
        clat = sum(f["latitude"] for f in facs) / len(facs)
        clon = sum(f["longitude"] for f in facs) / len(facs)
        centroids[rid] = (clat, clon)
    if not centroids or not facility_list:
        return []
    # Step 1: R_j = sum_i (pop_i * f(d_ij))
    R_j: List[float] = []
    for f in facility_list:
        s = 0.0
        for rid, (clat, clon) in centroids.items():
            d = haversine_km(clat, clon, f["latitude"], f["longitude"])
            pop = next((r["population"] for r in region_list if r["id"] == rid), 0) or 0
            s += pop * decay(d)
        R_j.append(s if s > 0 else 1.0)
    # Step 2: A_i = sum_j (cap_j * f(d_ij) / R_j)
    max_A = 0.0
    A_list: List[tuple[float, float, float, Optional[dict]]] = []
    for r in region_list:
        rid = r["id"]
        if rid not in centroids:
            continue
        clat, clon = centroids[rid]
        best_km = float("inf")
        best_fac: Optional[dict] = None
        A_i = 0.0
        for j, f in enumerate(facility_list):
            d = haversine_km(clat, clon, f["latitude"], f["longitude"])
            if d < best_km:
                best_km = d
                best_fac = f
            cap = f.get("bed_capacity") or 0
            A_i += cap * decay(d) / R_j[j]
        if A_i > max_A:
            max_A = A_i
        A_list.append((clat, clon, A_i, best_fac))
        if best_fac is None and facility_list:
            best_fac = facility_list[0]
            best_km = haversine_km(clat, clon, best_fac["latitude"], best_fac["longitude"])
    # Normalize to 0-1 and build AccessMetric
    scale = max_A if max_A > 0 else 1.0
    result: List[AccessMetric] = []
    region_subset = [x for x in region_list if x["id"] in centroids]
    for r, (clat, clon, A_i, best_fac) in zip(region_subset, A_list):
        rid = r["id"]
        score = A_i / scale
        d_km = haversine_km(clat, clon, best_fac["latitude"], best_fac["longitude"]) if best_fac else 0.0
        result.append(
            AccessMetric(
                region_id=rid,
                region_name=r["name"],
                population=r["population"],
                centroid_lat=round(clat, 6),
                centroid_lon=round(clon, 6),
                distance_km=round(d_km, 2),
                access_score=round(min(1.0, score), 4),
                nearest_facility_id=best_fac["id"] if best_fac else "",
                nearest_facility_name=best_fac["name"] if best_fac else "",
            )
        )
    return result


@app.get("/access/metrics", response_model=List[AccessMetric])
async def get_access_metrics() -> List[AccessMetric]:
    """Access metrics using nearest-facility model (default)."""
    pool = await get_pool()
    regions = await pool.fetch("SELECT id, name, country_code, population FROM regions ORDER BY id")
    facilities = await pool.fetch(
        "SELECT id, name, facility_type, region_id, latitude, longitude, bed_capacity FROM facilities ORDER BY id"
    )
    region_list = [dict(r) for r in regions]
    facility_list = [dict(f) for f in facilities]
    region_facilities: dict[str, list] = {}
    for f in facility_list:
        rid = f["region_id"]
        region_facilities.setdefault(rid, []).append(f)
    return _compute_nearest_metrics(region_list, facility_list, region_facilities)


@app.get("/access/summary", response_model=AccessSummaryResponse)
async def get_access_summary(model: str = "nearest") -> AccessSummaryResponse:
    """Access metrics with optional model (nearest | 2sfca) and equity (Gini)."""
    pool = await get_pool()
    regions = await pool.fetch("SELECT id, name, country_code, population FROM regions ORDER BY id")
    facilities = await pool.fetch(
        "SELECT id, name, facility_type, region_id, latitude, longitude, bed_capacity FROM facilities ORDER BY id"
    )
    region_list = [dict(r) for r in regions]
    facility_list = [dict(f) for f in facilities]
    region_facilities: dict[str, list] = {}
    for f in facility_list:
        rid = f["region_id"]
        region_facilities.setdefault(rid, []).append(f)
    if model == "2sfca":
        metrics = _compute_2sfca_metrics(region_list, facility_list, region_facilities)
    else:
        metrics = _compute_nearest_metrics(region_list, facility_list, region_facilities)
    equity_gini = _gini_coefficient(metrics)
    return AccessSummaryResponse(
        metrics=metrics,
        equity_gini=equity_gini,
        model=model if model == "2sfca" else "nearest",
    )


def _risk_level(utilization: float) -> str:
    if utilization > OVERLOAD_UTILIZATION_THRESHOLD:
        return "high"
    if utilization >= 0.7:
        return "medium"
    return "low"


@app.get("/demand/summary", response_model=List[DemandForecastSummary])
async def demand_summary(
    time_horizon_months: int = 1,
    facility_type: Optional[str] = None,
    overload_only: bool = False,
) -> List[DemandForecastSummary]:
    """
    Forecast and overload flag per facility. Optional filters: time_horizon_months (1/3/6/12),
    facility_type, overload_only. risk_level: low (<70% util), medium (70–100%), high (overload).
    """
    time_horizon_months = max(1, min(12, time_horizon_months))
    pool = await get_pool()
    rows = await pool.fetch(
        """
        SELECT
            f.id AS facility_id,
            f.name AS facility_name,
            f.facility_type,
            f.region_id,
            f.bed_capacity,
            COALESCE(AVG(d.visits)::float, 0) AS avg_monthly_visits
        FROM facilities f
        LEFT JOIN demand d ON d.facility_id = f.id
        GROUP BY f.id, f.name, f.facility_type, f.region_id, f.bed_capacity
        ORDER BY f.id
        """
    )

    summaries: List[DemandForecastSummary] = []
    for row in rows:
        data = dict(row)
        if facility_type and (data.get("facility_type") or "") != facility_type:
            continue
        capacity = data["bed_capacity"] or 0
        avg_visits = float(data["avg_monthly_visits"] or 0.0)
        forecast_next = avg_visits
        forecast_horizon = avg_visits * time_horizon_months
        utilization = (forecast_next / capacity) if capacity > 0 else 0.0
        overload = utilization > OVERLOAD_UTILIZATION_THRESHOLD if capacity > 0 else False
        if overload_only and not overload:
            continue
        risk_level = _risk_level(utilization)

        summaries.append(
            DemandForecastSummary(
                facility_id=data["facility_id"],
                facility_name=data["facility_name"],
                facility_type=(data.get("facility_type") or "unknown"),
                region_id=data["region_id"],
                bed_capacity=capacity,
                avg_monthly_visits=round(avg_visits, 2),
                forecast_next_month=round(forecast_next, 2),
                forecast_over_horizon=round(forecast_horizon, 2),
                utilization_ratio=round(utilization, 2),
                overload=overload,
                risk_score=round(utilization, 2),
                risk_level=risk_level,
            )
        )

    return summaries


@app.get("/demand/timeseries/{facility_id}", response_model=List[DemandPoint])
async def demand_timeseries(facility_id: str) -> List[DemandPoint]:
    """
    Return the historical demand time series for a single facility,
    ordered by date.
    """
    pool = await get_pool()
    rows = await pool.fetch(
        """
        SELECT date, visits
        FROM demand
        WHERE facility_id = $1
        ORDER BY date
        """,
        facility_id,
    )
    return [DemandPoint(**dict(row)) for row in rows]


@app.post("/scenarios", response_model=ScenarioDetail)
async def create_scenario(body: ScenarioCreate) -> ScenarioDetail:
    """Create and store a scenario (facility plan or mobile routes)."""
    pool = await get_pool()
    row = await pool.fetchrow(
        """INSERT INTO scenarios (name, description, type, inputs, results_summary)
           VALUES ($1, $2, $3, $4, $5)
           RETURNING id, name, description, type, created_by, created_at, inputs, results_summary""",
        body.name,
        body.description,
        body.type,
        body.inputs,
        body.results_summary,
    )
    if not row:
        raise HTTPException(status_code=500, detail="Failed to create scenario")
    sid = row["id"]
    if body.facility_results:
        for r in body.facility_results:
            await pool.execute(
                """INSERT INTO scenario_facility_results (scenario_id, region_id, region_name, centroid_lat, centroid_lon)
                   VALUES ($1, $2, $3, $4, $5)""",
                sid, r.region_id, r.region_name, r.centroid_lat, r.centroid_lon,
            )
    if body.route_results:
        for route in body.route_results:
            vid = route.get("vehicle_id", 0)
            stops = route.get("stops", [])
            prev_lat, prev_lon = None, None
            for s in stops:
                lat = s.get("latitude")
                lon = s.get("longitude")
                leg = None
                if prev_lat is not None and lat is not None and lon is not None:
                    leg = haversine_km(prev_lat, prev_lon, lat, lon)
                await pool.execute(
                    """INSERT INTO scenario_route_results (scenario_id, vehicle_id, stop_sequence, region_id, region_name, latitude, longitude, leg_distance_km)
                       VALUES ($1, $2, $3, $4, $5, $6, $7, $8)""",
                    sid, vid, s.get("sequence", 0), s.get("region_id", ""), s.get("region_name", ""),
                    lat, lon, leg,
                )
                prev_lat, prev_lon = lat, lon
    await audit_log("create_scenario", {"scenario_id": str(sid), "name": body.name})
    return await get_scenario_by_id(sid)


@app.get("/scenarios", response_model=List[ScenarioListItem])
async def list_scenarios() -> List[ScenarioListItem]:
    pool = await get_pool()
    rows = await pool.fetch(
        "SELECT id, name, description, type, created_at, results_summary FROM scenarios ORDER BY created_at DESC"
    )
    return [
        ScenarioListItem(
            id=r["id"],
            name=r["name"],
            description=r["description"],
            type=r["type"],
            created_at=r["created_at"],
            results_summary=r["results_summary"],
        )
        for r in rows
    ]


async def get_scenario_by_id(sid: UUID) -> ScenarioDetail:
    pool = await get_pool()
    row = await pool.fetchrow(
        "SELECT id, name, description, type, created_by, created_at, inputs, results_summary FROM scenarios WHERE id = $1",
        sid,
    )
    if not row:
        raise HTTPException(status_code=404, detail="Scenario not found")
    fac_rows = await pool.fetch(
        "SELECT region_id, region_name, centroid_lat, centroid_lon FROM scenario_facility_results WHERE scenario_id = $1",
        sid,
    )
    route_rows = await pool.fetch(
        "SELECT vehicle_id, stop_sequence, region_id, region_name, latitude, longitude, leg_distance_km FROM scenario_route_results WHERE scenario_id = $1 ORDER BY vehicle_id, stop_sequence",
        sid,
    )
    routes_by_v: dict[int, list] = {}
    for r in route_rows:
        v = int(r["vehicle_id"])
        if v not in routes_by_v:
            routes_by_v[v] = []
        routes_by_v[v].append({
            "region_id": r["region_id"],
            "region_name": r["region_name"],
            "latitude": r["latitude"],
            "longitude": r["longitude"],
            "sequence": r["stop_sequence"],
            "leg_distance_km": r["leg_distance_km"],
        })
    return ScenarioDetail(
        id=row["id"],
        name=row["name"],
        description=row["description"],
        type=row["type"],
        created_by=row["created_by"],
        created_at=row["created_at"],
        inputs=row["inputs"],
        results_summary=row["results_summary"],
        facility_results=[ScenarioFacilityResult(**dict(f)) for f in fac_rows],
        route_results=[{"vehicle_id": k, "stops": v} for k, v in sorted(routes_by_v.items())],
    )


@app.get("/scenarios/{scenario_id}", response_model=ScenarioDetail)
async def get_scenario(scenario_id: UUID) -> ScenarioDetail:
    return await get_scenario_by_id(scenario_id)


# --- CSV Export (Phase 7) ---


def _csv_stream(headers: List[str], rows: List[tuple]) -> io.StringIO:
    out = io.StringIO()
    w = csv.writer(out)
    w.writerow(headers)
    w.writerows(rows)
    out.seek(0)
    return out


def _csv_response(headers: List[str], rows: List[tuple], filename: str) -> Response:
    buf = _csv_stream(headers, rows)
    return Response(
        content=buf.getvalue().encode("utf-8"),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


@app.get("/export/regions")
async def export_regions_csv():
    pool = await get_pool()
    rows = await pool.fetch("SELECT id, name, country_code, population FROM regions ORDER BY id")
    data = [(r["id"], r["name"], r["country_code"], r["population"]) for r in rows]
    await audit_log("export_regions_csv")
    return _csv_response(["id", "name", "country_code", "population"], data, "regions.csv")


@app.get("/export/facilities")
async def export_facilities_csv():
    pool = await get_pool()
    rows = await pool.fetch("SELECT id, name, facility_type, region_id, latitude, longitude, bed_capacity FROM facilities ORDER BY id")
    data = [(r["id"], r["name"], r["facility_type"], r["region_id"], r["latitude"], r["longitude"], r["bed_capacity"]) for r in rows]
    await audit_log("export_facilities_csv")
    return _csv_response(["id", "name", "facility_type", "region_id", "latitude", "longitude", "bed_capacity"], data, "facilities.csv")


@app.get("/export/access")
async def export_access_csv():
    metrics = await get_access_metrics()
    data = [(m.region_id, m.region_name, m.population, m.centroid_lat, m.centroid_lon, m.distance_km, m.access_score, m.nearest_facility_id, m.nearest_facility_name) for m in metrics]
    await audit_log("export_access_csv")
    return _csv_response(
        ["region_id", "region_name", "population", "centroid_lat", "centroid_lon", "distance_km", "access_score", "nearest_facility_id", "nearest_facility_name"],
        data, "access_metrics.csv",
    )


@app.get("/export/demand")
async def export_demand_csv():
    summaries = await demand_summary()
    data = [(s.facility_id, s.facility_name, s.facility_type, s.region_id, s.bed_capacity, s.forecast_next_month, s.utilization_ratio, s.risk_level, s.overload) for s in summaries]
    await audit_log("export_demand_csv")
    return _csv_response(
        ["facility_id", "facility_name", "facility_type", "region_id", "bed_capacity", "forecast_next_month", "utilization_ratio", "risk_level", "overload"],
        data, "demand_summary.csv",
    )


def _compute_region_centroids(
    region_rows: List[dict], facility_rows: List[dict]
) -> dict:
    """Return mapping region_id -> (lat, lon) using facility centroids."""
    region_facilities: dict[str, list[dict]] = {}
    for f in facility_rows:
        rid = f["region_id"]
        region_facilities.setdefault(rid, []).append(f)

    centroids: dict[str, tuple[float, float]] = {}
    for r in region_rows:
        rid = r["id"]
        facs = region_facilities.get(rid, [])
        if not facs:
            continue
        clat = sum(f["latitude"] for f in facs) / len(facs)
        clon = sum(f["longitude"] for f in facs) / len(facs)
        centroids[rid] = (clat, clon)
    return centroids


def _population_weighted_distances(
    region_rows: List[dict],
    centroids: dict[str, tuple[float, float]],
    all_facility_points: List[tuple[float, float]],
) -> tuple[float, float]:
    """
    Compute population-weighted average distance and max distance
    from each region centroid to the nearest facility point.
    """
    total_pop = 0.0
    weighted_sum = 0.0
    max_dist = 0.0

    for r in region_rows:
        rid = r["id"]
        if rid not in centroids:
            continue
        clat, clon = centroids[rid]
        best_km = float("inf")
        for (lat, lon) in all_facility_points:
            d = haversine_km(clat, clon, lat, lon)
            if d < best_km:
                best_km = d
        pop = float(r["population"] or 0)
        total_pop += pop
        weighted_sum += pop * best_km
        if best_km > max_dist:
            max_dist = best_km

    avg = weighted_sum / total_pop if total_pop > 0 else 0.0
    return avg, max_dist


@app.get("/planning/facility", response_model=FacilityPlanResult)
async def facility_planning(
    num_new: int = 1,
    prioritize_underserved: bool = False,
) -> FacilityPlanResult:
    """
    Simple facility planning:
    - Treat each region centroid as a candidate new facility site.
    - Choose num_new sites that minimise population-weighted average
      distance (and, when prioritize_underserved=True, also favour
      reducing worst-region distance for equity).

    Brute-force search suitable for small numbers of regions (demo scale).
    """
    if num_new < 1:
        num_new = 1

    pool = await get_pool()
    regions = await pool.fetch("SELECT id, name, country_code, population FROM regions ORDER BY id")
    facilities = await pool.fetch(
        "SELECT id, name, facility_type, region_id, latitude, longitude, bed_capacity FROM facilities ORDER BY id"
    )
    region_list = [dict(r) for r in regions]
    facility_list = [dict(f) for f in facilities]

    if not region_list or not facility_list:
        return FacilityPlanResult(
            num_new=num_new,
            chosen_region_ids=[],
            before_avg_distance_km=0.0,
            after_avg_distance_km=0.0,
            before_max_distance_km=0.0,
            after_max_distance_km=0.0,
            worst_region_improvement_km=0.0,
        )

    centroids = _compute_region_centroids(region_list, facility_list)
    candidate_region_ids = list(centroids.keys())
    if not candidate_region_ids:
        return FacilityPlanResult(
            num_new=num_new,
            chosen_region_ids=[],
            before_avg_distance_km=0.0,
            after_avg_distance_km=0.0,
            before_max_distance_km=0.0,
            after_max_distance_km=0.0,
            worst_region_improvement_km=0.0,
        )

    existing_points = [(f["latitude"], f["longitude"]) for f in facility_list]
    before_avg, before_max = _population_weighted_distances(
        region_list, centroids, existing_points
    )

    num_new = max(1, min(num_new, len(candidate_region_ids)))

    from itertools import combinations

    best_choice: Optional[tuple[str, ...]] = None
    best_avg = float("inf")
    best_max = float("inf")

    # When prioritize_underserved, prefer solutions that reduce max distance (equity)
    alpha = 0.5 if prioritize_underserved else 0.0  # blend: 0.5*avg + 0.5*max

    for combo in combinations(candidate_region_ids, num_new):
        new_points = [centroids[rid] for rid in combo]
        all_points = existing_points + new_points
        avg, mx = _population_weighted_distances(region_list, centroids, all_points)
        score = (1 - alpha) * avg + alpha * mx if alpha > 0 else avg
        best_score = (1 - alpha) * best_avg + alpha * best_max if alpha > 0 else best_avg
        if score < best_score or (math.isclose(score, best_score) and mx < best_max):
            best_avg = avg
            best_max = mx
            best_choice = combo

    chosen_ids = list(best_choice) if best_choice else []
    improvement = before_max - (best_max if best_choice else before_max)

    return FacilityPlanResult(
        num_new=num_new,
        chosen_region_ids=chosen_ids,
        before_avg_distance_km=round(before_avg, 2),
        after_avg_distance_km=round(best_avg if best_choice else before_avg, 2),
        before_max_distance_km=round(before_max, 2),
        after_max_distance_km=round(best_max if best_choice else before_max, 2),
        worst_region_improvement_km=round(max(0.0, improvement), 2),
    )


# --- Mobile routing (VRP) ---


class RouteStop(BaseModel):
    region_id: str
    region_name: str
    latitude: float
    longitude: float
    sequence: int


class MobileRoute(BaseModel):
    vehicle_id: int
    stops: List[RouteStop]
    total_distance_km: float


class MobileRoutesResult(BaseModel):
    depot_lat: float
    depot_lon: float
    routes: List[MobileRoute]


@app.get("/planning/mobile-routes", response_model=MobileRoutesResult)
async def mobile_routes(
    num_vehicles: int = 2,
    region_ids: Optional[str] = None,
) -> MobileRoutesResult:
    """
    Simple VRP: assign region stops to vehicles starting from a depot (first facility).
    If region_ids is not provided, use all regions with access metrics (centroids).
    """
    if num_vehicles < 1:
        num_vehicles = 1

    pool = await get_pool()
    facilities = await pool.fetch(
        "SELECT id, name, facility_type, region_id, latitude, longitude FROM facilities ORDER BY id"
    )
    if not facilities:
        return MobileRoutesResult(
            depot_lat=0.0,
            depot_lon=0.0,
            routes=[],
        )

    depot_lat = float(facilities[0]["latitude"])
    depot_lon = float(facilities[0]["longitude"])

    regions = await pool.fetch("SELECT id, name FROM regions ORDER BY id")
    region_list = [dict(r) for r in regions]
    facility_list = [dict(f) for f in facilities]
    centroids = _compute_region_centroids(region_list, facility_list)
    region_name_by_id = {r["id"]: r["name"] for r in region_list}

    all_stops = [
        {
            "region_id": rid,
            "region_name": region_name_by_id.get(rid, rid),
            "lat": latlon[0],
            "lon": latlon[1],
        }
        for rid, latlon in centroids.items()
    ]

    if region_ids:
        wanted = {s.strip() for s in region_ids.split(",") if s.strip()}
        all_stops = [s for s in all_stops if s["region_id"] in wanted]
    if not all_stops:
        return MobileRoutesResult(depot_lat=depot_lat, depot_lon=depot_lon, routes=[])

    # Build distance matrix: index 0 = depot, 1..n = stops
    def dist_km(i: int, j: int) -> float:
        if i == 0 and j == 0:
            return 0.0
        if i == 0:
            return haversine_km(depot_lat, depot_lon, all_stops[j - 1]["lat"], all_stops[j - 1]["lon"])
        if j == 0:
            return haversine_km(all_stops[i - 1]["lat"], all_stops[i - 1]["lon"], depot_lat, depot_lon)
        return haversine_km(
            all_stops[i - 1]["lat"], all_stops[i - 1]["lon"],
            all_stops[j - 1]["lat"], all_stops[j - 1]["lon"],
        )

    n = len(all_stops)
    num_vehicles = min(num_vehicles, n)

    try:
        from ortools.constraint_solver import pywrapcp

        manager = pywrapcp.RoutingIndexManager(n + 1, num_vehicles, 0)
        routing = pywrapcp.RoutingModel(manager)

        def distance_callback(from_index: int, to_index: int) -> int:
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            return int(round(dist_km(from_node, to_node) * 1000))  # metres for integer

        transit_callback_index = routing.RegisterTransitCallback(distance_callback)
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
        routing.AddDimension(
            transit_callback_index,
            0,
            999999,
            True,
            "Distance",
        )
        dimension = routing.GetDimensionOrDie("Distance")

        search_params = pywrapcp.DefaultRoutingSearchParameters()
        search_params.first_solution_strategy = (
            pywrapcp.RoutingSearchParameters.FirstSolutionStrategy.PATH_CHEAPEST_ARC
        )

        solution = routing.SolveWithParameters(search_params)
        if not solution:
            # Fallback: greedy assign stops round-robin
            routes_out = []
            for v in range(num_vehicles):
                stops_for_v = [
                    all_stops[i]
                    for i in range(v, n, num_vehicles)
                ]
                total = 0.0
                prev_lat, prev_lon = depot_lat, depot_lon
                for s in stops_for_v:
                    total += haversine_km(prev_lat, prev_lon, s["lat"], s["lon"])
                    prev_lat, prev_lon = s["lat"], s["lon"]
                total += haversine_km(prev_lat, prev_lon, depot_lat, depot_lon)
                routes_out.append(
                    MobileRoute(
                        vehicle_id=v,
                        stops=[
                            RouteStop(
                                region_id=s["region_id"],
                                region_name=s["region_name"],
                                latitude=s["lat"],
                                longitude=s["lon"],
                                sequence=i + 1,
                            )
                            for i, s in enumerate(stops_for_v)
                        ],
                        total_distance_km=round(total, 2),
                    )
                )
            return MobileRoutesResult(
                depot_lat=depot_lat,
                depot_lon=depot_lon,
                routes=routes_out,
            )

        routes_out: List[MobileRoute] = []
        for vehicle_id in range(num_vehicles):
            index = routing.Start(vehicle_id)
            stops: List[dict] = []
            total_km = 0.0
            prev_node = 0
            while not routing.IsEnd(index):
                node = manager.IndexToNode(index)
                total_km += dist_km(prev_node, node)
                if node >= 1:
                    stops.append(all_stops[node - 1])
                prev_node = node
                index = solution.Value(routing.NextVar(index))
            total_km += dist_km(prev_node, 0)
            routes_out.append(
                MobileRoute(
                    vehicle_id=vehicle_id,
                    stops=[
                        RouteStop(
                            region_id=s["region_id"],
                            region_name=s["region_name"],
                            latitude=s["lat"],
                            longitude=s["lon"],
                            sequence=i + 1,
                        )
                        for i, s in enumerate(stops)
                    ],
                    total_distance_km=round(total_km, 2),
                )
            )
        return MobileRoutesResult(
            depot_lat=depot_lat,
            depot_lon=depot_lon,
            routes=routes_out,
        )
    except Exception:
        # No OR-Tools or solver failed: greedy round-robin
        routes_out = []
        for v in range(num_vehicles):
            stops_for_v = [all_stops[i] for i in range(v, n, num_vehicles)]
            total = 0.0
            prev_lat, prev_lon = depot_lat, depot_lon
            for s in stops_for_v:
                total += haversine_km(prev_lat, prev_lon, s["lat"], s["lon"])
                prev_lat, prev_lon = s["lat"], s["lon"]
            total += haversine_km(prev_lat, prev_lon, depot_lat, depot_lon)
            routes_out.append(
                MobileRoute(
                    vehicle_id=v,
                    stops=[
                        RouteStop(
                            region_id=s["region_id"],
                            region_name=s["region_name"],
                            latitude=s["lat"],
                            longitude=s["lon"],
                            sequence=i + 1,
                        )
                        for i, s in enumerate(stops_for_v)
                    ],
                    total_distance_km=round(total, 2),
                )
            )
        return MobileRoutesResult(
            depot_lat=depot_lat,
            depot_lon=depot_lon,
            routes=routes_out,
        )


# --- Admin (Phase 8) ---


class AuditLogEntry(BaseModel):
    id: UUID
    user_id: Optional[UUID]
    action_type: str
    details: Optional[dict]
    created_at: datetime


class ConfigEntry(BaseModel):
    key: str
    value: Any


class ProfileEntry(BaseModel):
    id: UUID
    email: Optional[str]
    role: str


@app.get("/admin/audit-logs", response_model=List[AuditLogEntry])
async def list_audit_logs(limit: int = 100) -> List[AuditLogEntry]:
    pool = await get_pool()
    rows = await pool.fetch(
        "SELECT id, user_id, action_type, details, created_at FROM audit_logs ORDER BY created_at DESC LIMIT $1",
        min(limit, 500),
    )
    return [
        AuditLogEntry(
            id=r["id"],
            user_id=r["user_id"],
            action_type=r["action_type"],
            details=r["details"],
            created_at=r["created_at"],
        )
        for r in rows
    ]


@app.get("/admin/config", response_model=List[ConfigEntry])
async def get_config() -> List[ConfigEntry]:
    pool = await get_pool()
    rows = await pool.fetch("SELECT key, value FROM config")
    return [ConfigEntry(key=r["key"], value=r["value"]) for r in rows]


@app.put("/admin/config")
async def set_config(key: str, value: str) -> dict:
    import json
    try:
        val = json.loads(value)
    except Exception:
        val = value
    pool = await get_pool()
    await pool.execute(
        "INSERT INTO config (key, value) VALUES ($1, $2) ON CONFLICT (key) DO UPDATE SET value = $2",
        key,
        val,
    )
    await audit_log("set_config", {"key": key})
    return {"ok": True}


@app.get("/admin/users", response_model=List[ProfileEntry])
async def list_users() -> List[ProfileEntry]:
    pool = await get_pool()
    rows = await pool.fetch("SELECT id, email, role FROM profiles ORDER BY email")
    return [
        ProfileEntry(id=r["id"], email=r["email"], role=r["role"])
        for r in rows
    ]


@app.post("/admin/upload/regions")
async def upload_regions_csv(file: UploadFile = File(...)) -> dict:
    content = (await file.read()).decode("utf-8")
    reader = csv.DictReader(io.StringIO(content))
    pool = await get_pool()
    n = 0
    for row in reader:
        id_ = row.get("id", "").strip()
        name = row.get("name", "").strip()
        country_code = row.get("country_code", "").strip()
        pop = row.get("population", "0").strip()
        population = int(pop) if pop.isdigit() else 0
        if not id_:
            continue
        await pool.execute(
            """INSERT INTO regions (id, name, country_code, population)
               VALUES ($1, $2, $3, $4)
               ON CONFLICT (id) DO UPDATE SET name = $2, country_code = $3, population = $4""",
            id_, name, country_code, population,
        )
        n += 1
    await audit_log("upload_regions_csv", {"rows": n})
    return {"uploaded": n}


@app.post("/admin/upload/facilities")
async def upload_facilities_csv(file: UploadFile = File(...)) -> dict:
    content = (await file.read()).decode("utf-8")
    reader = csv.DictReader(io.StringIO(content))
    pool = await get_pool()
    n = 0
    for row in reader:
        id_ = row.get("id", "").strip()
        name = row.get("name", "").strip()
        facility_type = row.get("facility_type", "clinic").strip()
        region_id = row.get("region_id", "").strip()
        lat = row.get("latitude", "0").strip()
        lon = row.get("longitude", "0").strip()
        cap = row.get("bed_capacity", "0").strip()
        lat_f = float(lat) if lat else 0.0
        lon_f = float(lon) if lon else 0.0
        bed_capacity = int(cap) if cap.isdigit() else 0
        if not id_ or not region_id:
            continue
        await pool.execute(
            """INSERT INTO facilities (id, name, facility_type, region_id, latitude, longitude, bed_capacity)
               VALUES ($1, $2, $3, $4, $5, $6, $7)
               ON CONFLICT (id) DO UPDATE SET name = $2, facility_type = $3, region_id = $4, latitude = $5, longitude = $6, bed_capacity = $7""",
            id_, name, facility_type, region_id, lat_f, lon_f, bed_capacity,
        )
        n += 1
    await audit_log("upload_facilities_csv", {"rows": n})
    return {"uploaded": n}


@app.post("/admin/upload/demand")
async def upload_demand_csv(file: UploadFile = File(...)) -> dict:
    content = (await file.read()).decode("utf-8")
    reader = csv.DictReader(io.StringIO(content))
    pool = await get_pool()
    n = 0
    for row in reader:
        date_s = row.get("date", "").strip()
        facility_id = row.get("facility_id", "").strip()
        visits_s = row.get("visits", "0").strip()
        if not date_s or not facility_id:
            continue
        try:
            from datetime import datetime as dt
            d = dt.fromisoformat(date_s.replace("Z", "+00:00")).date() if "T" in date_s else dt.strptime(date_s, "%Y-%m-%d").date()
        except Exception:
            continue
        visits = int(visits_s) if visits_s.isdigit() else 0
        await pool.execute(
            """INSERT INTO demand (date, facility_id, visits) VALUES ($1, $2, $3)""",
            d, facility_id, visits,
        )
        n += 1
    await audit_log("upload_demand_csv", {"rows": n})
    return {"uploaded": n}

