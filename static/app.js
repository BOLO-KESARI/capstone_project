/* ============================================================
   AviationStack Dashboard – Client-side JS
   ============================================================ */

(() => {
    "use strict";

    // ----- Element refs -----
    const endpointList   = document.getElementById("endpoint-list");
    const endpointTitle  = document.getElementById("endpoint-title");
    const filterBody     = document.getElementById("filter-body");
    const btnFetch       = document.getElementById("btn-fetch");
    const spinner        = document.getElementById("spinner");
    const errorAlert     = document.getElementById("error-alert");
    const resultsCard    = document.getElementById("results-card");
    const resultsThead   = document.getElementById("results-thead");
    const resultsTbody   = document.getElementById("results-tbody");
    const paginationBar  = document.getElementById("pagination-bar");
    const resultInfo     = document.getElementById("result-info");
    const btnPrev        = document.getElementById("btn-prev");
    const btnNext        = document.getElementById("btn-next");
    const jsonCard       = document.getElementById("json-card");
    const jsonBody       = document.getElementById("json-body");
    const jsonPre        = document.getElementById("json-pre");
    const btnToggleJson  = document.getElementById("btn-toggle-json");

    // ----- State -----
    let currentEndpoint = "flights";
    let currentOffset   = 0;
    let currentLimit    = 25;
    let totalResults    = 0;
    let lastJson        = null;
    
    // ----- Indian Airports List -----
    const INDIAN_AIRPORTS = [
        { val: "",    label: "— Select Airport —" },
        { val: "DEL", label: "DEL - Delhi" },
        { val: "BOM", label: "BOM - Mumbai" },
        { val: "BLR", label: "BLR - Bengaluru" },
        { val: "MAA", label: "MAA - Chennai" },
        { val: "HYD", label: "HYD - Hyderabad" },
        { val: "CCU", label: "CCU - Kolkata" },
        { val: "AMD", label: "AMD - Ahmedabad" },
        { val: "COK", label: "COK - Kochi" },
        { val: "PNQ", label: "PNQ - Pune" },
        { val: "GOI", label: "GOI - Goa (Dabolim)" },
        { val: "GOX", label: "GOX - Goa (Mopa)" },
        { val: "LKO", label: "LKO - Lucknow" },
        { val: "GAU", label: "GAU - Guwahati" },
        { val: "JAI", label: "JAI - Jaipur" },
        { val: "TRV", label: "TRV - Thiruvananthapuram" },
        { val: "BBI", label: "BBI - Bhubaneswar" },
        { val: "PAT", label: "PAT - Patna" },
        { val: "IXC", label: "IXC - Chandigarh" },
        { val: "IXJ", label: "IXJ - Jammu" },
        { val: "SXR", label: "SXR - Srinagar" },
        { val: "ATQ", label: "ATQ - Amritsar" },
        { val: "IDR", label: "IDR - Indore" },
        { val: "NAG", label: "NAG - Nagpur" },
        { val: "VTZ", label: "VTZ - Vishakhapatnam" }
    ];

    // ----- Endpoint filter definitions -----
    const ENDPOINT_FILTERS = {
        flights: [
            { key: "limit",         label: "Max Results",    type: "number", default: 25 },
            { key: "flight_status", label: "Status",         type: "select",
              options: ["", "scheduled", "active", "landed", "cancelled", "incident", "diverted"] },
            { key: "flight_date",   label: "Date",           type: "date" },
            { key: "dep_iata",      label: "Departure Airport", type: "select", options: INDIAN_AIRPORTS },
            { key: "arr_iata",      label: "Arrival Airport",   type: "select", options: INDIAN_AIRPORTS },
            { key: "airline_name",  label: "Airline Name",   type: "text" },
            { key: "airline_iata",  label: "Airline Code",   type: "text", placeholder: "e.g. AI" },
            { key: "flight_number", label: "Flight Number",  type: "text" },
        ],
        routes: [
            { key: "limit",         label: "Max Results",    type: "number", default: 25 },
            { key: "dep_iata",      label: "Departure Airport", type: "select", options: INDIAN_AIRPORTS },
            { key: "arr_iata",      label: "Arrival Airport",   type: "select", options: INDIAN_AIRPORTS },
            { key: "airline_iata",  label: "Airline Code",   type: "text" },
            { key: "flight_number", label: "Flight Number",  type: "text" },
        ],
        airports: [
            { key: "limit",  label: "Max Results",  type: "number", default: 25 },
            { key: "search", label: "Search Airport", type: "text",   placeholder: "Airport name or code" },
        ],
        airlines: [
            { key: "limit",  label: "Max Results",  type: "number", default: 25 },
            { key: "search", label: "Search Airline", type: "text",   placeholder: "Airline name or code" },
        ],
        airplanes: [
            { key: "limit",  label: "Max Results",  type: "number", default: 25 },
            { key: "search", label: "Search Planes", type: "text",   placeholder: "e.g. Boeing" },
        ],
        aircraft_types: [
            { key: "limit",  label: "Max Results",  type: "number", default: 25 },
            { key: "search", label: "Search Type",  type: "text" },
        ],
        taxes: [
            { key: "limit",  label: "Max Results",  type: "number", default: 25 },
            { key: "search", label: "Search Taxes", type: "text" },
        ],
        cities: [
            { key: "limit",  label: "Max Results",  type: "number", default: 25 },
            { key: "search", label: "City Name",    type: "text" },
        ],
        countries: [
            { key: "limit",  label: "Max Results",  type: "number", default: 25 },
            { key: "search", label: "Country Name", type: "text" },
        ],
        timetable: [
            { key: "limit",     label: "Max Results",     type: "number", default: 25 },
            { key: "iata_code", label: "Airport Code", type: "text",   placeholder: "e.g. JFK" },
            { key: "icao_code", label: "Global ID",    type: "text",   placeholder: "e.g. KJFK" },
        ],
        flights_future: [
            { key: "limit",     label: "Max Results",     type: "number", default: 25 },
            { key: "iata_code", label: "Airport Code", type: "text",   placeholder: "e.g. JFK" },
            { key: "icao_code", label: "Global ID",    type: "text" },
            { key: "date",      label: "Planned Date",    type: "date" },
        ],
        "sim/bookings": [
            { key: "limit", label: "Max Results", type: "number", default: 25 },
        ],
        "sim/flights": [
            { key: "limit", label: "Max Results", type: "number", default: 25 },
        ],
        "sim/passengers": [
            { key: "limit", label: "Max Results", type: "number", default: 25 },
        ],
        "sim/operations": [
            { key: "limit", label: "Max Results", type: "number", default: 25 },
        ],
        "sim/competitor": [
            { key: "limit", label: "Max Results", type: "number", default: 25 },
        ],
        "sim/events": [
            { key: "limit", label: "Max Results", type: "number", default: 25 },
        ],
        "sim/economy": [
            { key: "limit", label: "Max Results", type: "number", default: 25 },
        ],
        "sim/holidays": [
            { key: "limit", label: "Max Results", type: "number", default: 25 },
        ],
        "sim/traffic": [
            { key: "limit", label: "Max Results", type: "number", default: 25 },
        ],
        "sim/trends": [
            { key: "limit", label: "Max Results", type: "number", default: 25 },
        ],
        "sim/sentiment": [
            { key: "limit", label: "Max Results", type: "number", default: 25 },
        ],
        "sim/fuel": [
            { key: "limit", label: "Max Results", type: "number", default: 25 },
        ],
        "sim/booking_patterns": [
            { key: "limit", label: "Max Results", type: "number", default: 25 },
        ],
        "sim/advanced_analytics": [
            { key: "limit", label: "Max Results", type: "number", default: 25 },
        ],
        "weather": [
            { key: "iata_code", label: "Select Airport", type: "select", options: INDIAN_AIRPORTS.slice(1) },
        ]
    };

    // ----- Friendly column labels -----
    const NICE_LABELS = {
        flight_date: "Date", flight_status: "Status", airline: "Airline",
        departure: "Departure", arrival: "Arrival", flight: "Flight Info",
        airport_name: "Airport", iata_code: "Airport Code", icao_code: "Global ID",
        country_name: "Country", city_iata_code: "City Code",
        airline_name: "Airline Name", country_iso2: "Country ISO",
        iata_type: "Type", iata_prefix_accounting: "Prefix",
        registration_number: "Reg #", production_line: "Line",
        aircraft_name: "Aircraft Name", tax_name: "Tax Name", tax_id: "Tax ID",
        dep_iata: "Departure Airport", arr_iata: "Arrival Airport", airline_iata: "Airline Code",
        booking_id: "Booking ID", total_price_paid: "Price Paid", booking_status: "Status",
        actual_departure: "Actual Dep", actual_arrival: "Actual Arr", load_factor: "Load %",
        loyalty_tier: "Loyalty", churn_risk_score: "Churn Risk", event_name: "Event",
        competitor_price: "Comp. Price", competitor_market_share: "Comp. Share",
        temperature: "Temp", feels_like: "Feels Like", humidity: "Humidity",
        wind_speed: "Wind Speed", weather_condition: "Condition",
        visibility_km: "Visibility", severity_index: "Severity Index",
        gdp_growth_percent: "GDP Growth%", inflation_rate_percent: "Inflation%",
        unemployment_rate_percent: "Unemployment%", exchange_rate_usd_inr: "USD/INR",
        consumer_confidence_index: "CCI", aviation_sector_index: "Aviation Index",
        holiday_date: "Date", holiday_name: "Holiday", holiday_type: "Type",
        travel_demand_multiplier: "Demand Boost",
        total_departures: "Departs", total_arrivals: "Arrives",
        average_departure_delay: "Avg Delay(m)", runway_utilization_percent: "Runway %",
        congestion_index: "Congestion Index",
        search_keyword: "Keyword", search_volume_index: "SVI",
        trend_growth_rate: "Growth Rate", region: "Region",
        airline: "Airline", post_date: "Date", platform: "Platform",
        sentiment_score: "Sentiment", complaint_category: "Complaint",
        engagement_score: "Likes/Engagement",
        fuel_type: "Type", price_per_litre: "Price/Litre",
        price_change_percent: "Volatility",
        velocity_idx: "Booking Velocity", cum_30d: "30D Cumul.",
        cum_7d: "7D Cumul.", pace_vs_historical: "Pace vs Hist",
        abandonment_rate: "Abandon %", conversion_rate: "Conv %",
        top_booking_dow: "Peak Day", peak_booking_time: "Peak Time",
        haul_category: "Haul Type", fare_index_vs_route: "Fare Index",
        yield: "Yield/Pax", rasm: "RASM", price_position: "Vs Comp",
        biz_travel_score: "Biz Score", estimated_load_factor: "Est. Load%",
        booking_velocity_7d: "7D Velocity"
    };

    // ----- Render filter controls -----
    function renderFilters(endpoint) {
        const filters = ENDPOINT_FILTERS[endpoint] || [];
        filterBody.innerHTML = "";
        if (!filters.length) {
            filterBody.innerHTML = `<p class="text-muted mb-0">No filters for this endpoint.</p>`;
            return;
        }
        const row = document.createElement("div");
        row.className = "row g-3";
        filters.forEach((f) => {
            const col = document.createElement("div");
            col.className = "col-md-4 col-lg-3";
            let input = "";
            if (f.type === "select") {
                const opts = f.options
                    .map((o) => {
                        const val = typeof o === 'object' ? o.val : o;
                        const label = typeof o === 'object' ? o.label : (o || "— Any —");
                        return `<option value="${val}">${label}</option>`;
                    })
                    .join("");
                input = `<select class="form-select form-select-sm" id="f-${f.key}">${opts}</select>`;
            } else if (f.type === "date") {
                input = `<input type="date" class="form-control form-control-sm" id="f-${f.key}" />`;
            } else if (f.type === "number") {
                input = `<input type="number" class="form-control form-control-sm" id="f-${f.key}" value="${f.default || ""}" min="1" max="100" />`;
            } else {
                input = `<input type="text" class="form-control form-control-sm" id="f-${f.key}" placeholder="${f.placeholder || ""}" />`;
            }
            col.innerHTML = `<label class="form-label" for="f-${f.key}">${f.label}</label>${input}`;
            row.appendChild(col);
        });
        filterBody.appendChild(row);
    }

    // ----- Collect filter values -----
    function collectFilters() {
        const filters = ENDPOINT_FILTERS[currentEndpoint] || [];
        const params = {};
        filters.forEach((f) => {
            const el = document.getElementById(`f-${f.key}`);
            if (el && el.value) {
                params[f.key] = el.value;
            }
        });
        // override limit
        if (params.limit) currentLimit = parseInt(params.limit, 10);
        params.offset = currentOffset;
        return params;
    }

    // ----- Fetch data -----
    async function fetchData(resetOffset = true) {
        if (resetOffset) currentOffset = 0;
        const params = collectFilters();
        const qs = new URLSearchParams(params).toString();
        const url = `/api/${currentEndpoint}?${qs}`;

        // UI state
        spinner.classList.remove("d-none");
        errorAlert.classList.add("d-none");
        resultsCard.style.display = "none";
        jsonCard.style.display = "none";

        try {
            const res = await fetch(url);
            if (!res.ok) throw new Error(`HTTP ${res.status} – ${res.statusText}`);
            const json = await res.json();
            lastJson = json;

            // check API-level error
            if (json.error) {
                throw new Error(json.error.message || JSON.stringify(json.error));
            }

            const data = json.data || [];
            const pagination = json.pagination || {};
            totalResults = pagination.total || data.length;

            renderTable(data);
            updatePagination(pagination);
            jsonPre.textContent = JSON.stringify(json, null, 2);
            jsonCard.style.display = "block";
        } catch (err) {
            errorAlert.textContent = err.message;
            errorAlert.classList.remove("d-none");
        } finally {
            spinner.classList.add("d-none");
        }
    }

    // ----- Render table -----
    function renderTable(data) {
        resultsThead.innerHTML = "";
        resultsTbody.innerHTML = "";
        if (!data.length) {
            resultsCard.style.display = "block";
            resultsTbody.innerHTML = `<tr><td class="text-center text-muted py-4">No results found.</td></tr>`;
            return;
        }

        // Flatten first-level nested objects for table display
        const flatRows = data.map(flattenRow);
        const keys = Object.keys(flatRows[0]);

        // thead
        const tr = document.createElement("tr");
        keys.forEach((k) => {
            const th = document.createElement("th");
            th.textContent = NICE_LABELS[k] || k.replace(/_/g, " ");
            tr.appendChild(th);
        });
        resultsThead.appendChild(tr);

        // tbody
        flatRows.forEach((row) => {
            const tr = document.createElement("tr");
            keys.forEach((k) => {
                const td = document.createElement("td");
                let val = row[k];
                // status badge
                if (k === "flight_status" && val) {
                    const cls = `status-${val}`;
                    td.innerHTML = `<span class="badge badge-status ${cls}">${val}</span>`;
                } else {
                    td.textContent = val ?? "—";
                }
                tr.appendChild(td);
            });
            resultsTbody.appendChild(tr);
        });
        resultsCard.style.display = "block";
    }

    // ----- Flatten a row -----
    function flattenRow(obj) {
        const flat = {};
        for (const [k, v] of Object.entries(obj)) {
            if (v && typeof v === "object" && !Array.isArray(v)) {
                // one level deep – prefix with parent key
                for (const [k2, v2] of Object.entries(v)) {
                    if (typeof v2 !== "object") {
                        flat[`${k}_${k2}`] = v2;
                    }
                }
            } else if (Array.isArray(v)) {
                flat[k] = v.length ? JSON.stringify(v) : "—";
            } else {
                flat[k] = v;
            }
        }
        return flat;
    }

    // ----- Pagination -----
    function updatePagination(pagination) {
        paginationBar.style.display = "flex";
        const from = currentOffset + 1;
        const to = Math.min(currentOffset + currentLimit, totalResults);
        resultInfo.textContent = `Showing ${from}–${to} of ${totalResults} results`;
        btnPrev.disabled = currentOffset === 0;
        btnNext.disabled = currentOffset + currentLimit >= totalResults;
    }

    // ----- Event listeners -----
    endpointList.addEventListener("click", (e) => {
        const btn = e.target.closest("[data-endpoint]");
        if (!btn) return;
        endpointList.querySelectorAll(".list-group-item").forEach((b) => b.classList.remove("active"));
        btn.classList.add("active");
        currentEndpoint = btn.dataset.endpoint;
        endpointTitle.textContent = btn.textContent.trim();
        currentOffset = 0;
        renderFilters(currentEndpoint);
        // hide previous results
        resultsCard.style.display = "none";
        jsonCard.style.display = "none";
        paginationBar.style.display = "none";
        errorAlert.classList.add("d-none");
    });

    btnFetch.addEventListener("click", () => fetchData(true));

    btnPrev.addEventListener("click", () => {
        currentOffset = Math.max(0, currentOffset - currentLimit);
        fetchData(false);
    });
    btnNext.addEventListener("click", () => {
        currentOffset += currentLimit;
        fetchData(false);
    });

    btnToggleJson.addEventListener("click", () => {
        jsonBody.classList.toggle("d-none");
    });

    // Allow Enter key in filter inputs to trigger fetch
    filterBody.addEventListener("keydown", (e) => {
        if (e.key === "Enter") fetchData(true);
    });

    // ----- Init -----
    renderFilters(currentEndpoint);
})();
