from __future__ import annotations

import math
import uuid
from datetime import datetime, date
from typing import Any, Dict, List, Optional

from fastapi import (
    Depends,
    FastAPI,
    File,
    HTTPException,
    UploadFile,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from dotenv import load_dotenv

from app.db import get_db
from app.auth import get_current_user, get_current_user_optional

load_dotenv()


# --------------------------------------------------------------------------------------
# In‑memory demo data
# --------------------------------------------------------------------------------------

REGIONS: List[Dict[str, Any]] = [
    {"id": "R1", "name": "Central District", "country_code": "IN", "population": 250_000, "lat": 12.9716, "lon": 77.5946},
    {"id": "R2", "name": "North District", "country_code": "IN", "population": 180_000, "lat": 13.0500, "lon": 77.5800},
    {"id": "R3", "name": "East District", "country_code": "IN", "population": 320_000, "lat": 12.9850, "lon": 77.7000},
    {"id": "R4", "name": "South District", "country_code": "IN", "population": 150_000, "lat": 12.8800, "lon": 77.5800},
    {"id": "R5", "name": "West District", "country_code": "IN", "population": 210_000, "lat": 12.9900, "lon": 77.4800},
]

FACILITIES: List[Dict[str, Any]] = [
    {
        "id": "F1",
        "name": "Central General Hospital",
        "facility_type": "hospital",
        "region_id": "R1",
        "latitude": 12.9716,
        "longitude": 77.5946,
        "bed_capacity": 300,
    },
    {
        "id": "F2",
        "name": "North Community Clinic",
        "facility_type": "clinic",
        "region_id": "R2",
        "latitude": 13.0500,
        "longitude": 77.5800,
        "bed_capacity": 40,
    },
    {
        "id": "F3",
        "name": "East Referral Hospital",
        "facility_type": "hospital",
        "region_id": "R3",
        "latitude": 12.9850,
        "longitude": 77.7000,
        "bed_capacity": 220,
    },
    {
        "id": "F4",
        "name": "South Primary Health Center",
        "facility_type": "clinic",
        "region_id": "R4",
        "latitude": 12.8800,
        "longitude": 77.5800,
        "bed_capacity": 25,
    },
    {
        "id": "F5",
        "name": "West District Hospital",
        "facility_type": "hospital",
        "region_id": "R5",
        "latitude": 12.9900,
        "longitude": 77.4800,
        "bed_capacity": 180,
    },
]

# Simple monthly demand (visits) per facility for last 12 months
DEMAND_TIMESERIES: Dict[str, List[Dict[str, Any]]] = {}

today = date.today()
for facility in FACILITIES:
    fid = facility["id"]
    series: List[Dict[str, Any]] = []
    base = facility["bed_capacity"] * 0.8
    for i in range(12, 0, -1):
        month = date(today.year if today.month - i > 0 else today.year - 1, ((today.month - i - 1) % 12) + 1, 1)
        # small seasonal variation
        visits = base + (i - 6) * 5
        series.append({"date": month.isoformat(), "visits": max(int(visits), 1)})
    DEMAND_TIMESERIES[fid] = series

# Scenarios and audit logs kept in memory (ok for demo / local use)
SCENARIOS: Dict[str, Dict[str, Any]] = {}
AUDIT_LOGS: List[Dict[str, Any]] = []


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------

def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great‑circle distance between two points on Earth."""
    r = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return round(r * c, 2)


def compute_access_metrics(model: str = "nearest") -> List[Dict[str, Any]]:
    """Compute basic access metrics for each region."""
    metrics: List[Dict[str, Any]] = []
    if not REGIONS or not FACILITIES:
        return metrics

    for region in REGIONS:
        distances: List[Dict[str, Any]] = []
        for fac in FACILITIES:
            d = haversine_km(region["lat"], region["lon"], fac["latitude"], fac["longitude"])
            distances.append({"facility": fac, "distance_km": d})
        distances.sort(key=lambda x: x["distance_km"])
        nearest = distances[0]
        distance_km = nearest["distance_km"]

        if model == "2sfca":
            # crude 2SFCA‑like score: shorter distance and more beds = better
            score = 1.0 / (1.0 + distance_km) * (nearest["facility"]["bed_capacity"] / 300.0)
        else:
            score = 1.0 / (1.0 + distance_km)

        metrics.append(
            {
                "region_id": region["id"],
                "region_name": region["name"],
                "population": region["population"],
                "centroid_lat": region["lat"],
                "centroid_lon": region["lon"],
                "distance_km": distance_km,
                "access_score": round(score, 4),
                "nearest_facility_id": nearest["facility"]["id"],
                "nearest_facility_name": nearest["facility"]["name"],
            }
        )
    return metrics


def gini(values: List[float]) -> float:
    if not values:
        return 0.0
    sorted_vals = sorted(v for v in values if v >= 0)
    n = len(sorted_vals)
    if n == 0:
        return 0.0
    cum = 0.0
    for i, v in enumerate(sorted_vals, start=1):
        cum += i * v
    total = sum(sorted_vals)
    return round((2 * cum) / (n * total) - (n + 1) / n, 4) if total > 0 else 0.0


def append_audit(user_id: Optional[str], action_type: str, details: Dict[str, Any]) -> None:
    AUDIT_LOGS.insert(
        0,
        {
            "id": str(uuid.uuid4()),
            "user_id": user_id,
            "action_type": action_type,
            "details": details,
            "created_at": datetime.utcnow().isoformat() + "Z",
        },
    )
    # keep last 200
    if len(AUDIT_LOGS) > 200:
        AUDIT_LOGS.pop()


def demand_summary_internal(time_horizon_months: int = 6) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for fac in FACILITIES:
        series = DEMAND_TIMESERIES.get(fac["id"], [])
        if not series:
            continue
        avg_monthly = sum(p["visits"] for p in series) / len(series)
        forecast_next = avg_monthly
        forecast_horizon = avg_monthly * time_horizon_months
        capacity = fac["bed_capacity"] * time_horizon_months
        utilization = forecast_horizon / capacity if capacity > 0 else 0.0
        overload = utilization > 1.0
        risk_level = "high" if utilization > 1.2 else "medium" if utilization > 0.9 else "low"
        rows.append(
            {
                "facility_id": fac["id"],
                "facility_name": fac["name"],
                "facility_type": fac["facility_type"],
                "region_id": fac["region_id"],
                "bed_capacity": fac["bed_capacity"],
                "avg_monthly_visits": round(avg_monthly, 1),
                "forecast_next_month": round(forecast_next, 1),
                "forecast_over_horizon": round(forecast_horizon, 1),
                "utilization_ratio": round(utilization, 3),
                "overload": overload,
                "risk_score": round(utilization, 3),
                "risk_level": risk_level,
            }
        )
    return rows


# --------------------------------------------------------------------------------------
# FastAPI app
# --------------------------------------------------------------------------------------

app = FastAPI(title="Healthcare Access API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok", "message": "API running"}


@app.get("/auth/me")
def auth_me(user: dict = Depends(get_current_user)):
    user_id = user["sub"]
    email = user.get("email") or ""
    role = "viewer"

    # Try to read role from profiles table if it exists
    try:
        with get_db() as conn:
            cur = conn.cursor()
            cur.execute(
                "SELECT id, email, role FROM public.profiles WHERE id = %s::uuid",
                (user_id,),
            )
            row = cur.fetchone()
            cur.close()
        if row:
            email = row.get("email") or email
            role = row.get("role") or role
    except Exception:
        # Fail soft if profiles table is missing or query fails
        pass

    return {
        "user_id": user_id,
        "email": email,
        "role": role,
    }


# --------------------------------------------------------------------------------------
# Regions & facilities
# --------------------------------------------------------------------------------------


@app.get("/regions")
def list_regions(_: Optional[dict] = Depends(get_current_user_optional)):
    return REGIONS


@app.get("/facilities")
def list_facilities(_: Optional[dict] = Depends(get_current_user_optional)):
    return FACILITIES


# --------------------------------------------------------------------------------------
# Dashboard & access
# --------------------------------------------------------------------------------------


@app.get("/dashboard/summary")
def dashboard_summary(_: Optional[dict] = Depends(get_current_user_optional)):
    metrics = compute_access_metrics("nearest")
    distances = [m["distance_km"] for m in metrics]
    total_pop = sum(r["population"] for r in REGIONS) or 0

    # consider adequate if distance <= 10 km
    adequate_regions = [m for m in metrics if m["distance_km"] <= 10]
    adequate_pop = sum(
        next(r["population"] for r in REGIONS if r["id"] == m["region_id"])
        for m in adequate_regions
    ) if adequate_regions else 0
    pct_adequate = round(100 * adequate_pop / total_pop, 1) if total_pop else 0.0

    demand_rows = demand_summary_internal(6)
    overloaded = [r for r in demand_rows if r["overload"]]

    alerts: List[str] = []
    for row in overloaded:
        alerts.append(
            f"Facility {row['facility_name']} is forecast to be overloaded (utilization {int(row['utilization_ratio']*100)}%)."
        )
    for m in metrics:
        if m["distance_km"] > 25:
            alerts.append(f"Region {m['region_name']} has very poor access (distance {m['distance_km']} km).")

    return {
        "total_population": total_pop,
        "population_adequate_access": adequate_pop,
        "pct_adequate": pct_adequate,
        "underserved_region_count": len([m for m in metrics if m["distance_km"] > 15]),
        "overloaded_facility_count": len(overloaded),
        "alerts": alerts,
        "last_updated_regions": None,
        "last_updated_facilities": None,
        "last_updated_demand": None,
    }


@app.get("/access/summary")
def access_summary(model: str = "nearest", _: Optional[dict] = Depends(get_current_user_optional)):
    if model not in {"nearest", "2sfca"}:
        model = "nearest"
    metrics = compute_access_metrics(model)
    scores = [m["access_score"] for m in metrics]
    equity_gini = gini(scores)
    return {"metrics": metrics, "equity_gini": equity_gini, "model": model}


@app.get("/access/metrics")
def access_metrics(_: Optional[dict] = Depends(get_current_user_optional)):
    return compute_access_metrics("nearest")


# --------------------------------------------------------------------------------------
# Demand & overload
# --------------------------------------------------------------------------------------


@app.get("/demand/summary")
def demand_summary(
    time_horizon_months: int = 6,
    facility_type: str = "",
    overload_only: bool = False,
    _: Optional[dict] = Depends(get_current_user_optional),
):
    rows = demand_summary_internal(time_horizon_months)
    if facility_type:
        rows = [r for r in rows if r["facility_type"] == facility_type]
    if overload_only:
        rows = [r for r in rows if r["overload"]]
    return rows


@app.get("/demand/timeseries/{facility_id}")
def demand_timeseries(facility_id: str, _: Optional[dict] = Depends(get_current_user_optional)):
    return DEMAND_TIMESERIES.get(facility_id, [])


# --------------------------------------------------------------------------------------
# Facility planning
# --------------------------------------------------------------------------------------


@app.get("/planning/facility")
def planning_facility(
    num_new: int = 1,
    prioritize_underserved: bool = False,
    _: Optional[dict] = Depends(get_current_user_optional),
):
    num_new = max(1, min(num_new, len(REGIONS)))
    base_metrics = compute_access_metrics("nearest")
    before_avg = round(sum(m["distance_km"] for m in base_metrics) / len(base_metrics), 2) if base_metrics else 0.0
    before_max = max((m["distance_km"] for m in base_metrics), default=0.0)

    # choose regions with worst access (largest distance / lowest score)
    sorted_regions = sorted(
        base_metrics,
        key=lambda m: (-m["distance_km"], m["access_score"] if prioritize_underserved else 0),
    )
    chosen = sorted_regions[:num_new]
    chosen_ids = [c["region_id"] for c in chosen]

    # assume new facilities built at chosen region centroids and recompute distances
    extended_facilities = FACILITIES + [
        {
            "id": f"NEW_{rid}",
            "name": f"New facility in {c['region_name']}",
            "facility_type": "hospital",
            "region_id": rid,
            "latitude": c["centroid_lat"],
            "longitude": c["centroid_lon"],
            "bed_capacity": 150,
        }
        for rid, c in zip(chosen_ids, chosen)
    ]

    # recompute access with extended facilities
    metrics_after: List[Dict[str, Any]] = []
    for region in REGIONS:
        distances = [
            haversine_km(region["lat"], region["lon"], f["latitude"], f["longitude"])
            for f in extended_facilities
        ]
        if not distances:
            continue
        dmin = min(distances)
        metrics_after.append({"region_id": region["id"], "distance_km": dmin})

    after_avg = round(
        sum(m["distance_km"] for m in metrics_after) / len(metrics_after),
        2,
    ) if metrics_after else before_avg
    after_max = max((m["distance_km"] for m in metrics_after), default=before_max)

    worst_before = max(base_metrics, key=lambda m: m["distance_km"]) if base_metrics else None
    worst_after = max(metrics_after, key=lambda m: m["distance_km"]) if metrics_after else None
    worst_improvement = 0.0
    if worst_before and worst_after and worst_before["region_id"] == worst_after["region_id"]:
        worst_improvement = round(worst_before["distance_km"] - worst_after["distance_km"], 2)

    return {
        "num_new": num_new,
        "chosen_region_ids": chosen_ids,
        "before_avg_distance_km": before_avg,
        "after_avg_distance_km": after_avg,
        "before_max_distance_km": before_max,
        "after_max_distance_km": after_max,
        "worst_region_improvement_km": worst_improvement,
    }


@app.get("/planning/mobile-routes")
def planning_mobile_routes(num_vehicles: int = 2, _: Optional[dict] = Depends(get_current_user_optional)):
    num_vehicles = max(1, min(num_vehicles, len(REGIONS)))
    if not FACILITIES or not REGIONS:
        return {"depot_lat": 0.0, "depot_lon": 0.0, "routes": []}

    depot = FACILITIES[0]
    depot_lat, depot_lon = depot["latitude"], depot["longitude"]

    # simple round‑robin assignment of regions to vehicles (excluding depot region if present)
    target_regions = REGIONS.copy()
    routes: List[Dict[str, Any]] = [{"vehicle_id": i, "stops": [], "total_distance_km": 0.0} for i in range(num_vehicles)]

    for idx, region in enumerate(target_regions):
        vehicle_idx = idx % num_vehicles
        routes[vehicle_idx]["stops"].append(
            {
                "region_id": region["id"],
                "region_name": region["name"],
                "latitude": region["lat"],
                "longitude": region["lon"],
                "sequence": len(routes[vehicle_idx]["stops"]) + 1,
            }
        )

    # compute total distance for each route (depot → stops → depot)
    for route in routes:
        points = [(depot_lat, depot_lon)] + [(s["latitude"], s["longitude"]) for s in route["stops"]] + [
            (depot_lat, depot_lon)
        ]
        total = 0.0
        for a, b in zip(points, points[1:]):
            total += haversine_km(a[0], a[1], b[0], b[1])
        route["total_distance_km"] = round(total, 2)

    return {"depot_lat": depot_lat, "depot_lon": depot_lon, "routes": routes}


# --------------------------------------------------------------------------------------
# Scenarios
# --------------------------------------------------------------------------------------


@app.get("/scenarios")
def list_scenarios(_: Optional[dict] = Depends(get_current_user_optional)):
    return [
        {
            "id": sid,
            "name": s["name"],
            "description": s.get("description"),
            "type": s["type"],
            "created_at": s["created_at"],
            "results_summary": s.get("results_summary"),
        }
        for sid, s in SCENARIOS.items()
    ]


@app.get("/scenarios/{scenario_id}")
def get_scenario(scenario_id: str, _: Optional[dict] = Depends(get_current_user_optional)):
    scenario = SCENARIOS.get(scenario_id)
    if not scenario:
        raise HTTPException(status_code=404, detail="Scenario not found")
    return scenario


@app.post("/scenarios")
def create_scenario(payload: Dict[str, Any], user: Optional[dict] = Depends(get_current_user_optional)):
    sid = str(uuid.uuid4())
    scenario = {
        "id": sid,
        "name": payload.get("name", "Untitled scenario"),
        "description": payload.get("description"),
        "type": payload.get("type", "facility_plan"),
        "created_at": datetime.utcnow().isoformat() + "Z",
        "created_by": user.get("sub"),
        "results_summary": payload.get("results_summary"),
        "facility_results": payload.get("facility_results"),
        "route_results": payload.get("route_results"),
    }
    SCENARIOS[sid] = scenario
    user_id = user.get("sub") if user else None
    append_audit(user_id, "create_scenario", {"scenario_id": sid, "type": scenario["type"]})
    return {"id": sid}


# --------------------------------------------------------------------------------------
# Admin: audit logs, users, uploads
# --------------------------------------------------------------------------------------


@app.get("/admin/audit-logs")
def admin_audit_logs(_: Optional[dict] = Depends(get_current_user_optional)):
    return AUDIT_LOGS


@app.get("/admin/users")
def admin_users(_: Optional[dict] = Depends(get_current_user_optional)):
    users: List[Dict[str, Any]] = []
    try:
        with get_db() as conn:
            cur = conn.cursor()
            cur.execute("SELECT id, email, role FROM public.profiles ORDER BY email NULLS LAST")
            for row in cur.fetchall():
                users.append({"id": str(row["id"]), "email": row.get("email"), "role": row.get("role", "viewer")})
            cur.close()
    except Exception:
        # soft‑fail if profiles table does not exist
        pass
    return users


@app.post("/admin/upload/regions")
def upload_regions(file: UploadFile = File(...), user: Optional[dict] = Depends(get_current_user_optional)):
    content = file.file.read().decode("utf-8")
    import csv

    reader = csv.DictReader(content.splitlines())
    required_cols = {"id", "name", "country_code", "population"}
    if not required_cols.issubset(reader.fieldnames or []):
        raise HTTPException(status_code=400, detail=f"CSV must contain columns: {', '.join(sorted(required_cols))}")

    new_regions: List[Dict[str, Any]] = []
    for row in reader:
        try:
            new_regions.append(
                {
                    "id": row["id"],
                    "name": row["name"],
                    "country_code": row["country_code"],
                    "population": int(row["population"] or 0),
                    "lat": float(row.get("lat") or 0.0),
                    "lon": float(row.get("lon") or 0.0),
                }
            )
        except ValueError:
            continue

    if new_regions:
        REGIONS.clear()
        REGIONS.extend(new_regions)
    user_id = user.get("sub") if user else None
    append_audit(user_id, "upload_regions", {"uploaded": len(new_regions)})
    return {"uploaded": len(new_regions)}


@app.post("/admin/upload/facilities")
def upload_facilities(file: UploadFile = File(...), user: Optional[dict] = Depends(get_current_user_optional)):
    content = file.file.read().decode("utf-8")
    import csv

    reader = csv.DictReader(content.splitlines())
    required_cols = {"id", "name", "facility_type", "region_id", "latitude", "longitude", "bed_capacity"}
    if not required_cols.issubset(reader.fieldnames or []):
        raise HTTPException(status_code=400, detail=f"CSV must contain columns: {', '.join(sorted(required_cols))}")

    new_facilities: List[Dict[str, Any]] = []
    for row in reader:
        try:
            new_facilities.append(
                {
                    "id": row["id"],
                    "name": row["name"],
                    "facility_type": row["facility_type"],
                    "region_id": row["region_id"],
                    "latitude": float(row["latitude"]),
                    "longitude": float(row["longitude"]),
                    "bed_capacity": int(row["bed_capacity"] or 0),
                }
            )
        except ValueError:
            continue

    if new_facilities:
        FACILITIES.clear()
        FACILITIES.extend(new_facilities)
    user_id = user.get("sub") if user else None
    append_audit(user_id, "upload_facilities", {"uploaded": len(new_facilities)})
    return {"uploaded": len(new_facilities)}


@app.post("/admin/upload/demand")
def upload_demand(file: UploadFile = File(...), user: Optional[dict] = Depends(get_current_user_optional)):
    content = file.file.read().decode("utf-8")
    import csv

    reader = csv.DictReader(content.splitlines())
    required_cols = {"date", "facility_id", "visits"}
    if not required_cols.issubset(reader.fieldnames or []):
        raise HTTPException(status_code=400, detail=f"CSV must contain columns: {', '.join(sorted(required_cols))}")

    # reset timeseries
    for fid in list(DEMAND_TIMESERIES.keys()):
        DEMAND_TIMESERIES[fid] = []

    count = 0
    for row in reader:
        fid = row["facility_id"]
        if fid not in DEMAND_TIMESERIES:
            DEMAND_TIMESERIES[fid] = []
        try:
            DEMAND_TIMESERIES[fid].append({"date": row["date"], "visits": int(row["visits"] or 0)})
            count += 1
        except ValueError:
            continue

    # sort by date for each facility
    for series in DEMAND_TIMESERIES.values():
        series.sort(key=lambda p: p["date"])

    user_id = user.get("sub") if user else None
    append_audit(user_id, "upload_demand", {"rows": count})
    return {"uploaded": count}


# --------------------------------------------------------------------------------------
# CSV exports
# --------------------------------------------------------------------------------------


def _csv_response(filename: str, header: List[str], rows: List[List[Any]]) -> StreamingResponse:
    import csv
    from io import StringIO

    buf = StringIO()
    writer = csv.writer(buf)
    writer.writerow(header)
    for row in rows:
        writer.writerow(row)
    buf.seek(0)
    return StreamingResponse(
        buf,
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.get("/export/regions")
def export_regions(_: Optional[dict] = Depends(get_current_user_optional)):
    header = ["id", "name", "country_code", "population", "lat", "lon"]
    rows = [
        [r["id"], r["name"], r["country_code"], r["population"], r.get("lat"), r.get("lon")]
        for r in REGIONS
    ]
    return _csv_response("regions.csv", header, rows)


@app.get("/export/facilities")
def export_facilities(_: Optional[dict] = Depends(get_current_user_optional)):
    header = ["id", "name", "facility_type", "region_id", "latitude", "longitude", "bed_capacity"]
    rows = [
        [
            f["id"],
            f["name"],
            f["facility_type"],
            f["region_id"],
            f["latitude"],
            f["longitude"],
            f["bed_capacity"],
        ]
        for f in FACILITIES
    ]
    return _csv_response("facilities.csv", header, rows)


@app.get("/export/access")
def export_access(_: Optional[dict] = Depends(get_current_user_optional)):
    metrics = compute_access_metrics("nearest")
    header = [
        "region_id",
        "region_name",
        "population",
        "centroid_lat",
        "centroid_lon",
        "distance_km",
        "access_score",
        "nearest_facility_id",
        "nearest_facility_name",
    ]
    rows = [
        [
            m["region_id"],
            m["region_name"],
            m["population"],
            m["centroid_lat"],
            m["centroid_lon"],
            m["distance_km"],
            m["access_score"],
            m["nearest_facility_id"],
            m["nearest_facility_name"],
        ]
        for m in metrics
    ]
    return _csv_response("access_metrics.csv", header, rows)


@app.get("/export/demand")
def export_demand(_: Optional[dict] = Depends(get_current_user_optional)):
    header = ["facility_id", "date", "visits"]
    rows: List[List[Any]] = []
    for fid, series in DEMAND_TIMESERIES.items():
        for p in series:
            rows.append([fid, p["date"], p["visits"]])
    return _csv_response("demand_summary.csv", header, rows)
import traceback
from fastapi import Request
from fastapi.responses import JSONResponse

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    print("\n\n========= BACKEND CRASH =========")
    traceback.print_exc()
    print("=================================\n\n")
    return JSONResponse(
        status_code=503,
        content={"error": str(exc), "path": str(request.url)}
    )
