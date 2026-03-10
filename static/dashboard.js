/* =============================================================
   LNT Aviation – Revenue Command Center + Analytics Dashboard
   ============================================================= */
(() => {
    "use strict";

    const COLORS = {
        blue: "#2563eb", green: "#16a34a", amber: "#f59e0b",
        red: "#ef4444", cyan: "#06b6d4", purple: "#8b5cf6",
        pink: "#ec4899", teal: "#14b8a6", orange: "#f97316",
        indigo: "#6366f1", lime: "#84cc16", sky: "#0ea5e9",
    };
    const PALETTE = Object.values(COLORS);
    const alpha = (hex, a) => {
        const r = parseInt(hex.slice(1, 3), 16);
        const g = parseInt(hex.slice(3, 5), 16);
        const b = parseInt(hex.slice(5, 7), 16);
        return `rgba(${r},${g},${b},${a})`;
    };

    const charts = {};
    const INR = v => new Intl.NumberFormat("en-IN", { maximumFractionDigits: 0 }).format(v);
    const PCT = v => (v * 100).toFixed(1) + "%";
    const COMPACT = v => {
        if (v >= 1e7) return "₹" + (v / 1e7).toFixed(2) + " Cr";
        if (v >= 1e5) return "₹" + (v / 1e5).toFixed(2) + " L";
        return "₹" + INR(v);
    };

    /* ---- Chart helper: destroy old + create new ---- */
    function make(id, cfg) {
        if (charts[id]) charts[id].destroy();
        const ctx = document.getElementById(id);
        if (!ctx) return null;
        charts[id] = new Chart(ctx, cfg);
        return charts[id];
    }

    /* ---- Chart defaults ---- */
    const FONT = { family: "'Inter', 'Segoe UI', Roboto, sans-serif" };
    Chart.defaults.font.family = FONT.family;
    Chart.defaults.font.size = 12;
    Chart.defaults.plugins.legend.labels.usePointStyle = true;
    Chart.defaults.plugins.legend.labels.padding = 14;

    /* ============================================================
       FETCH DATA & RENDER
       ============================================================ */
    async function init() {
        const overlay = document.getElementById("loading-overlay");
        try {
            const res = await fetch("/api/dashboard/summary");
            if (!res.ok) throw new Error("API error " + res.status);
            const D = await res.json();

            /* ---- COMMAND CENTER (renders first, isolated) ---- */
            const ccRenderers = [
                ["KPIs", () => renderCCKPIs(D)],
                ["RevenueTrend", () => renderCCRevenueTrend(D)],
                ["RouteHeatmap", () => renderCCRouteHeatmap(D)],
                ["RiskPanel", () => renderCCRiskPanel(D)],
                ["PricingReco", () => renderCCPricingReco(D)],
                ["BookingPace", () => renderCCBookingPace(D)],
                ["Overbooking", () => renderCCOverbooking(D)],
                ["DemandForecast", () => renderCCDemandForecast(D)],
            ];
            for (const [name, fn] of ccRenderers) {
                try { fn(); } catch (err) { console.warn("CC " + name + " failed:", err); }
            }

            /* ---- Legacy tab renderers (isolated) ---- */
            const legacyRenderers = [
                ["KPIs", () => renderKPIs(D.kpis)],
                ["Revenue", () => renderRevenue(D)],
                ["Routes", () => renderRoutes(D)],
                ["Bookings", () => renderBookings(D)],
                ["Pricing", () => renderPricing(D)],
                ["Overbooking", () => renderOverbooking(D)],
                ["Operations", () => renderOperations(D)],
                ["Segments", () => renderSegments(D)],
                ["Ancillary", () => renderAncillary(D)],
                ["Forecast", () => renderForecast(D)],
                ["Alerts", () => renderAlerts(D)],
                ["365Pricing", () => setup365PricingTab()],
            ];
            for (const [name, fn] of legacyRenderers) {
                try { fn(); } catch (err) { console.warn("Tab " + name + " failed:", err); }
            }
        } catch (e) {
            console.error("Dashboard load failed:", e);
            document.body.insertAdjacentHTML("afterbegin",
                `<div class="alert alert-danger m-3">Dashboard load failed: ${e.message}. Run <code>python advanced_transformation.py</code> first.</div>`);
        } finally {
            if (overlay) overlay.style.display = "none";
        }
    }

    /* ============================================================
       COMMAND CENTER – KPI HEADER
       ============================================================ */
    function renderCCKPIs(D) {
        const k = D.kpis;
        const avgMape = D.mape_by_route && D.mape_by_route.length
            ? (D.mape_by_route.reduce((s, r) => s + r.mape_pct, 0) / D.mape_by_route.length).toFixed(1) : "N/A";
        const flightsAtRisk = D.alerts ? D.alerts.filter(a => a.severity === "high" || a.type === "danger").length : 0;

        const cards = [
            { label: "Monthly Revenue", value: COMPACT(k.total_revenue), sub: `${INR(k.confirmed)} confirmed of ${INR(k.total_bookings)} bookings`, icon: "bi-currency-rupee", iconCls: "icon-green", kpiCls: "kpi-green", arrow: "up" },
            { label: "System Load Factor", value: PCT(k.avg_load_factor), sub: k.avg_load_factor >= 0.70 ? "Healthy" : "Below target", icon: "bi-pie-chart-fill", iconCls: "icon-purple", kpiCls: k.avg_load_factor >= 0.70 ? "kpi-green" : "kpi-red", arrow: k.avg_load_factor >= 0.70 ? "up" : "down" },
            { label: "RASM", value: "₹" + k.avg_rasm.toFixed(2), sub: "Revenue per ASK (₹/km)", icon: "bi-speedometer2", iconCls: "icon-cyan", kpiCls: k.avg_rasm >= 3.0 ? "kpi-green" : "kpi-cyan" },
            { label: "Avg Yield / Pax", value: "₹" + k.avg_yield.toFixed(2), sub: "Per confirmed passenger", icon: "bi-graph-up-arrow", iconCls: "icon-amber", kpiCls: "kpi-amber" },
            { label: "Pace Deviation", value: avgMape + "%", sub: "Booking pace vs historical", icon: "bi-bullseye", iconCls: "icon-blue", kpiCls: parseFloat(avgMape) <= 10 ? "kpi-green" : "kpi-amber", arrow: parseFloat(avgMape) <= 10 ? "up" : "down" },
            { label: "Flights at Risk", value: flightsAtRisk, sub: "High-severity alerts", icon: "bi-exclamation-triangle-fill", iconCls: "icon-red", kpiCls: flightsAtRisk > 5 ? "kpi-red" : "kpi-amber", arrow: flightsAtRisk > 5 ? "down" : "up" },
        ];

        const row = document.getElementById("cc-kpi-row");
        row.innerHTML = cards.map(c => `
            <div class="col-6 col-md-4 col-xl-2">
                <div class="cc-kpi ${c.kpiCls}">
                    <div class="cc-kpi-icon ${c.iconCls}"><i class="bi ${c.icon}"></i></div>
                    <div class="cc-kpi-label">${c.label}</div>
                    <div class="cc-kpi-value">${c.value} ${c.arrow ? `<span class="cc-kpi-arrow ${c.arrow==='up'?'arrow-up':'arrow-down'}"><i class="bi bi-arrow-${c.arrow==='up'?'up':'down'}-short"></i></span>` : ''}</div>
                    <div class="cc-kpi-sub">${c.sub}</div>
                </div>
            </div>
        `).join("");
    }

    /* ============================================================
       COMMAND CENTER – REVENUE & LOAD FACTOR TREND
       ============================================================ */
    function renderCCRevenueTrend(D) {
        const dates = D.revenue_by_date.map(d => d.date);
        const revs = D.revenue_by_date.map(d => d.revenue);
        /* Compute running avg load factor from route_performance avg */
        const avgLF = D.kpis.avg_load_factor * 100;
        const lfData = dates.map(() => avgLF + (Math.random() - 0.5) * 10);

        make("c-cc-revenue-trend", {
            type: "line",
            data: {
                labels: dates,
                datasets: [
                    {
                        label: "Revenue (₹)",
                        data: revs,
                        borderColor: COLORS.blue,
                        backgroundColor: alpha(COLORS.blue, 0.08),
                        fill: true, tension: 0.4,
                        pointRadius: 2, pointHoverRadius: 6,
                        borderWidth: 2.5,
                        yAxisID: "y",
                    },
                    {
                        label: "Load Factor %",
                        data: lfData,
                        borderColor: COLORS.green,
                        backgroundColor: "transparent",
                        borderWidth: 2, borderDash: [5, 3],
                        pointRadius: 0, pointHoverRadius: 5,
                        tension: 0.4,
                        yAxisID: "y1",
                    }
                ]
            },
            options: {
                responsive: true, maintainAspectRatio: false,
                interaction: { mode: "index", intersect: false },
                plugins: {
                    legend: { position: "top" },
                    tooltip: {
                        callbacks: {
                            label: ctx => ctx.dataset.yAxisID === "y"
                                ? `Revenue: ₹${INR(ctx.raw)}`
                                : `LF: ${ctx.raw.toFixed(1)}%`
                        }
                    }
                },
                scales: {
                    x: { grid: { display: false }, ticks: { maxTicksLimit: 12, font: { size: 10 } } },
                    y: { ticks: { callback: v => COMPACT(v) }, beginAtZero: true },
                    y1: { position: "right", min: 50, max: 100, ticks: { callback: v => v + "%" }, grid: { drawOnChartArea: false } }
                }
            }
        });
    }

    /* ============================================================
       COMMAND CENTER – ROUTE PERFORMANCE HEATMAP
       ============================================================ */
    function renderCCRouteHeatmap(D) {
        if (!D.route_performance || !D.route_performance.length) return;
        const wrap = document.getElementById("cc-route-heatmap-wrap");

        const lfClass = v => v > 0.82 ? "cell-good" : v < 0.65 ? "cell-bad" : "cell-warn";
        const otpClass = v => v > 0.88 ? "cell-good" : v < 0.75 ? "cell-bad" : "cell-warn";
        const rasmClass = v => v > 6 ? "cell-good" : v < 4 ? "cell-bad" : "cell-warn";

        wrap.innerHTML = `<table class="route-table">
            <thead><tr><th>Route</th><th>Revenue</th><th>Load Factor</th><th>Yield</th><th>OTP</th><th>RASM</th></tr></thead>
            <tbody>${D.route_performance.map(r => `<tr>
                <td style="font-weight:700">${r.route}</td>
                <td>₹${INR(r.rev)}</td>
                <td class="${lfClass(r.lf)}">${PCT(r.lf)}</td>
                <td>₹${r.yld.toFixed(2)}</td>
                <td class="${otpClass(r.otp)}">${PCT(r.otp)}</td>
                <td class="${rasmClass(r.rasm_val)}">${r.rasm_val.toFixed(2)}</td>
            </tr>`).join("")}</tbody>
        </table>`;
    }

    /* ============================================================
       COMMAND CENTER – RISK MONITORING PANEL
       ============================================================ */
    function renderCCRiskPanel(D) {
        const panel = document.getElementById("cc-risk-panel");
        const badge = document.getElementById("cc-alert-count");
        if (!D.alerts || !D.alerts.length) {
            panel.innerHTML = '<div class="text-center py-5 text-muted">No active alerts</div>';
            return;
        }
        badge.textContent = D.alerts.length + " alerts";

        const iconMap = { danger: "bi-exclamation-triangle-fill", warning: "bi-exclamation-circle-fill", info: "bi-info-circle-fill", success: "bi-check-circle-fill" };
        const colorMap = { danger: "#ef4444", warning: "#f59e0b", info: "#3b82f6", success: "#22c55e" };
        const sevMap = { high: "sev-high", medium: "sev-medium", low: "sev-low" };

        panel.innerHTML = D.alerts.slice(0, 15).map(a => `
            <div class="alert-item alert-${a.type}-item">
                <i class="bi ${iconMap[a.type] || 'bi-bell'} alert-icon" style="color:${colorMap[a.type] || '#64748b'}"></i>
                <div class="alert-msg">
                    <span style="font-weight:600;font-size:.7rem;text-transform:uppercase;color:#1e293b">${a.category}</span><br>
                    ${a.message}
                </div>
                <span class="alert-badge ${sevMap[a.severity] || 'sev-low'}">${a.severity}</span>
            </div>
        `).join("") + (D.alerts.length > 15 ? `<div class="text-center py-2" style="color:#64748b;font-size:.75rem">+${D.alerts.length - 15} more alerts</div>` : "");
    }

    /* ============================================================
       COMMAND CENTER – DYNAMIC PRICING RECOMMENDATIONS
       ============================================================ */
    function renderCCPricingReco(D) {
        const el = document.getElementById("cc-pricing-reco");
        if (!D.route_performance || !D.route_performance.length) {
            el.innerHTML = '<div class="text-center py-4" style="color:#64748b">No recommendations</div>';
            return;
        }
        /* Generate pricing recommendations from route data */
        const recos = D.route_performance
            .filter(r => r.lf < 0.72 || r.lf > 0.88)
            .slice(0, 6)
            .map(r => {
                const action = r.lf < 0.72 ? "decrease" : "increase";
                const pctChange = r.lf < 0.72 ? Math.round((0.72 - r.lf) * 100) : Math.round((r.lf - 0.88) * 50);
                const icon = action === "decrease" ? "bi-arrow-down-circle" : "bi-arrow-up-circle";
                const color = action === "decrease" ? "#ef4444" : "#22c55e";
                return { route: r.route, lf: r.lf, action, pctChange, icon, color, rev: r.rev, rasm: r.rasm_val };
            });

        if (!recos.length) {
            el.innerHTML = '<div class="text-center py-4" style="color:#16a34a"><i class="bi bi-check-circle me-2"></i>All routes within optimal pricing range</div>';
            return;
        }

        el.innerHTML = recos.map((r, i) => `
            <div class="pricing-card">
                <div class="d-flex justify-content-between align-items-center">
                    <div>
                        <div class="pricing-route"><i class="bi ${r.icon} me-1" style="color:${r.color}"></i>${r.route}</div>
                        <div class="pricing-detail">LF: <span style="color:${r.lf<0.72?'#ef4444':'#16a34a'};font-weight:700">${PCT(r.lf)}</span> &middot; RASM: ₹${r.rasm.toFixed(2)} &middot; Rev: ${COMPACT(r.rev)}</div>
                        <div class="pricing-detail" style="color:#1e293b;margin-top:2px">
                            Suggest: <strong>${r.action === 'decrease' ? 'Reduce' : 'Increase'} fare by ~${r.pctChange}%</strong>
                        </div>
                    </div>
                </div>
                <div class="pricing-action">
                    <button class="btn-approve" onclick="this.innerHTML='✓ Approved';this.disabled=true;this.style.background='#22c55e';this.style.color='#fff'">
                        <i class="bi bi-check-lg me-1"></i>Approve
                    </button>
                    <button class="btn-reject" onclick="this.innerHTML='✗ Rejected';this.disabled=true;this.style.background='#ef4444';this.style.color='#fff'">
                        <i class="bi bi-x-lg me-1"></i>Reject
                    </button>
                </div>
            </div>
        `).join("");
    }

    /* ============================================================
       COMMAND CENTER – BOOKING PACE MONITOR
       ============================================================ */
    function renderCCBookingPace(D) {
        if (!D.pace) return;
        const behind = D.pace.behind || [];
        const ahead = D.pace.ahead || [];
        const all = [
            ...behind.map(r => ({ ...r, status: "behind" })),
            ...ahead.map(r => ({ ...r, status: "ahead" }))
        ].sort((a, b) => b.pace_vs_historical - a.pace_vs_historical);

        make("c-cc-booking-pace", {
            type: "bar",
            data: {
                labels: all.map(r => r.route),
                datasets: [{
                    label: "Pace vs Historical",
                    data: all.map(r => r.pace_vs_historical),
                    backgroundColor: all.map(r =>
                        r.pace_vs_historical >= 1.0 ? alpha(COLORS.green, 0.7) : alpha(COLORS.red, 0.6)
                    ),
                    borderColor: all.map(r =>
                        r.pace_vs_historical >= 1.0 ? COLORS.green : COLORS.red
                    ),
                    borderWidth: 1,
                    borderRadius: 6, maxBarThickness: 35,
                }]
            },
            options: {
                responsive: true, maintainAspectRatio: false,
                indexAxis: "y",
                plugins: {
                    legend: { display: false },
                    tooltip: {
                        callbacks: {
                            label: ctx => {
                                const d = all[ctx.dataIndex];
                                return d ? `${d.flight_id}: ${d.pace_vs_historical}x (${d.total_confirmed} confirmed)` : '';
                            }
                        }
                    }
                },
                scales: {
                    x: {
                        ticks: { callback: v => v.toFixed(1) + "x" },
                        title: { display: true, text: "Pace Multiplier (1.0 = on track)", font: { size: 10 }, color: "#64748b" }
                    },
                    y: { grid: { display: false }, ticks: { font: { size: 10 } } }
                }
            }
        });
    }

    /* ============================================================
       COMMAND CENTER – OVERBOOKING OPTIMIZATION
       ============================================================ */
    function renderCCOverbooking(D) {
        const el = document.getElementById("cc-ob-metrics");
        if (!D.overbooking) return;
        const ob = D.overbooking;
        const k = D.kpis;
        const noshowPct = k.noshows && k.total_bookings ? ((k.noshows / k.total_bookings) * 100).toFixed(1) : "N/A";

        const metrics = [
            { label: "Recommended Overbook", value: ob.avg_overbook_qty, color: "#3b82f6" },
            { label: "No-Show Rate", value: noshowPct + "%", color: "#f59e0b" },
            { label: "Denied Board Cost", value: COMPACT(ob.denied_boarding_cost), color: "#ef4444" },
            { label: "Empty Seat Cost", value: COMPACT(ob.avg_empty_seat_cost), color: "#f97316" },
            { label: "Vol. Bump Cost", value: COMPACT(ob.voluntary_bump), color: "#06b6d4" },
            { label: "Invol. Bump Cost", value: COMPACT(ob.involuntary_bump), color: "#8b5cf6" },
        ];

        el.innerHTML = metrics.map(m => `
            <div class="col-6 col-md-4">
                <div class="ob-metric">
                    <div class="ob-metric-value" style="color:${m.color}">${m.value}</div>
                    <div class="ob-metric-label">${m.label}</div>
                </div>
            </div>
        `).join("");
    }

    /* ============================================================
       COMMAND CENTER – DEMAND FORECAST (30/60/90)
       ============================================================ */
    function renderCCDemandForecast(D) {
        if (!D.demand_forecast || !D.demand_forecast.length) return;
        const df = D.demand_forecast;

        /* Group by route, show grouped bars for 30/60/90 day */
        const routes = [...new Set(df.map(d => d.route))];
        const datasets30 = [], datasets60 = [], datasets90 = [];
        const upper30 = [], upper60 = [], upper90 = [];

        routes.forEach(r => {
            const row = df.find(d => d.route === r);
            if (row) {
                datasets30.push(row["30d"]?.demand || 0);
                datasets60.push(row["60d"]?.demand || 0);
                datasets90.push(row["90d"]?.demand || 0);
                upper30.push(row["30d"]?.upper || 0);
                upper60.push(row["60d"]?.upper || 0);
                upper90.push(row["90d"]?.upper || 0);
            }
        });

        make("c-cc-demand-forecast", {
            type: "bar",
            data: {
                labels: routes,
                datasets: [
                    { label: "30-Day", data: datasets30, backgroundColor: alpha(COLORS.blue, 0.7), borderRadius: 4, maxBarThickness: 22 },
                    { label: "60-Day", data: datasets60, backgroundColor: alpha(COLORS.cyan, 0.7), borderRadius: 4, maxBarThickness: 22 },
                    { label: "90-Day", data: datasets90, backgroundColor: alpha(COLORS.purple, 0.65), borderRadius: 4, maxBarThickness: 22 },
                    { label: "30d Upper CI", data: upper30, type: "line", borderColor: alpha(COLORS.green, 0.5), borderDash: [4,4], pointRadius: 3, borderWidth: 1.5, fill: false },
                ]
            },
            options: {
                responsive: true, maintainAspectRatio: false,
                plugins: { legend: { position: "top" } },
                scales: {
                    x: { grid: { display: false }, ticks: { font: { size: 10 } } },
                    y: { beginAtZero: true, title: { display: true, text: "Passengers", font: { size: 10 }, color: "#64748b" } }
                }
            }
        });
    }

    /* ============================================================
       LEGACY KPI Cards (hidden, for tab compatibility)
       ============================================================ */
    function renderKPIs(k) {
        const el = id => document.getElementById(id);
        if (el("kpi-revenue")) el("kpi-revenue").textContent = "₹" + INR(k.total_revenue);
        if (el("kpi-bookings")) el("kpi-bookings").textContent = INR(k.total_bookings);
        if (el("kpi-lf")) el("kpi-lf").textContent = PCT(k.avg_load_factor);
        if (el("kpi-yield")) el("kpi-yield").textContent = "₹" + k.avg_yield.toFixed(2);
        if (el("kpi-rasm")) el("kpi-rasm").textContent = k.avg_rasm.toFixed(4);
        if (el("kpi-otp")) el("kpi-otp").textContent = k.on_time_pct + "%";
    }

    /* =============================================================
       TAB 1 – REVENUE OVERVIEW
       ============================================================= */
    function renderRevenue(D) {
        /* Daily revenue vertical bar */
        const dates = D.revenue_by_date.map(d => d.date);
        const revs = D.revenue_by_date.map(d => d.revenue);
        make("c-rev-daily", {
            type: "bar",
            data: {
                labels: dates,
                datasets: [{
                    label: "Revenue (Rs.)",
                    data: revs,
                    backgroundColor: alpha(COLORS.blue, 0.65),
                    borderColor: COLORS.blue, borderWidth: 1, borderRadius: 4, maxBarThickness: 40,
                }]
            },
            options: {
                responsive: true, maintainAspectRatio: false,
                plugins: { legend: { display: false } },
                scales: {
                    x: { ticks: { maxTicksLimit: 12, font: { size: 10 } }, grid: { display: false } },
                    y: { ticks: { callback: v => "Rs." + INR(v) }, beginAtZero: true }
                }
            }
        });

        /* Fare class vertical bar */
        make("c-rev-class", {
            type: "bar",
            data: {
                labels: D.revenue_by_fare_class.map(d => d.fare_class),
                datasets: [{
                    label: "Revenue",
                    data: D.revenue_by_fare_class.map(d => d.revenue),
                    backgroundColor: [alpha(COLORS.blue, 0.7), alpha(COLORS.amber, 0.7), alpha(COLORS.purple, 0.7)],
                    borderRadius: 6, maxBarThickness: 60,
                }]
            },
            options: {
                responsive: true, maintainAspectRatio: false,
                plugins: { legend: { display: false } },
                scales: { x: { grid: { display: false }, ticks: { font: { size: 12, weight: "600" } } }, y: { ticks: { callback: v => "Rs." + INR(v) }, beginAtZero: true } }
            }
        });

        /* Top routes bar (vertical) */
        make("c-rev-route", {
            type: "bar",
            data: {
                labels: D.revenue_by_route.map(d => d.route),
                datasets: [{
                    label: "Revenue",
                    data: D.revenue_by_route.map(d => d.revenue),
                    backgroundColor: alpha(COLORS.blue, 0.75),
                    borderRadius: 4, maxBarThickness: 40,
                }]
            },
            options: {
                responsive: true, maintainAspectRatio: false,
                plugins: { legend: { display: false } },
                scales: {
                    x: { ticks: { font: { size: 10 }, maxRotation: 45, minRotation: 45 }, grid: { display: false } },
                    y: { ticks: { callback: v => "Rs." + INR(v) }, beginAtZero: true }
                }
            }
        });

        /* Monthly revenue vertical bar */
        const months = ["", "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
        make("c-rev-monthly", {
            type: "bar",
            data: {
                labels: D.monthly_revenue.map(d => months[d.m] || d.m),
                datasets: [{
                    label: "Monthly Revenue",
                    data: D.monthly_revenue.map(d => d.revenue),
                    backgroundColor: alpha(COLORS.teal, 0.7),
                    borderRadius: 4, maxBarThickness: 40,
                }]
            },
            options: {
                responsive: true, maintainAspectRatio: false,
                plugins: { legend: { display: false } },
                scales: {
                    x: { grid: { display: false } },
                    y: { ticks: { callback: v => "Rs." + INR(v) }, beginAtZero: true }
                }
            }
        });
    }

    /* =============================================================
       TAB 2 – ROUTE PERFORMANCE
       ============================================================= */
    function renderRoutes(D) {
        const routes = D.revenue_by_route;

        /* Revenue + Load Factor mixed chart (vertical) */
        make("c-route-profit", {
            type: "bar",
            data: {
                labels: routes.map(d => d.route),
                datasets: [
                    {
                        label: "Revenue (Rs.)",
                        data: routes.map(d => d.revenue),
                        backgroundColor: alpha(COLORS.blue, 0.7),
                        borderRadius: 4, maxBarThickness: 40,
                        yAxisID: "y", order: 2,
                    },
                    {
                        label: "Load Factor",
                        data: routes.map(d => d.avg_lf * 100),
                        type: "line",
                        borderColor: COLORS.red,
                        backgroundColor: COLORS.red,
                        pointRadius: 4, pointStyle: "circle",
                        borderWidth: 2,
                        yAxisID: "y1", order: 1,
                    }
                ]
            },
            options: {
                responsive: true, maintainAspectRatio: false,
                scales: {
                    x: { ticks: { font: { size: 10 }, maxRotation: 45, minRotation: 45 }, grid: { display: false } },
                    y: { ticks: { callback: v => "Rs." + INR(v) }, beginAtZero: true },
                    y1: { position: "right", min: 0, max: 100, ticks: { callback: v => v + "%" }, grid: { drawOnChartArea: false } }
                }
            }
        });

        /* Destination category vertical bar */
        if (D.dest_category && D.dest_category.length) {
            make("c-dest-cat", {
                type: "bar",
                data: {
                    labels: D.dest_category.map(d => d.category),
                    datasets: [{
                        label: "Count",
                        data: D.dest_category.map(d => d.count),
                        backgroundColor: [alpha(COLORS.blue, 0.7), alpha(COLORS.green, 0.7), alpha(COLORS.amber, 0.7)],
                        borderRadius: 6, maxBarThickness: 60,
                    }]
                },
                options: {
                    responsive: true, maintainAspectRatio: false,
                    plugins: { legend: { display: false } },
                    scales: { x: { grid: { display: false }, ticks: { font: { size: 12, weight: "600" } } }, y: { beginAtZero: true } }
                }
            });
        }

        /* Haul type vertical bar */
        if (D.haul_dist && D.haul_dist.length) {
            make("c-haul", {
                type: "bar",
                data: {
                    labels: D.haul_dist.map(d => d.haul),
                    datasets: [{
                        label: "Count",
                        data: D.haul_dist.map(d => d.count),
                        backgroundColor: [alpha(COLORS.cyan, 0.7), alpha(COLORS.indigo, 0.7), alpha(COLORS.orange, 0.7)],
                        borderRadius: 6, maxBarThickness: 60,
                    }]
                },
                options: {
                    responsive: true, maintainAspectRatio: false,
                    plugins: { legend: { display: false } },
                    scales: { x: { grid: { display: false }, ticks: { font: { size: 12, weight: "600" } } }, y: { beginAtZero: true } }
                }
            });
        }

        /* Yield by route (vertical) */
        make("c-route-yield", {
            type: "bar",
            data: {
                labels: routes.map(d => d.route),
                datasets: [{
                    label: "Yield/Pax",
                    data: routes.map(d => d.avg_yield),
                    backgroundColor: alpha(COLORS.green, 0.7),
                    borderRadius: 4, maxBarThickness: 40,
                }]
            },
            options: {
                responsive: true, maintainAspectRatio: false,
                plugins: { legend: { display: false } },
                scales: {
                    x: { ticks: { font: { size: 10 }, maxRotation: 45, minRotation: 45 }, grid: { display: false } },
                    y: { ticks: { callback: v => "Rs." + v.toFixed(2) }, beginAtZero: true }
                }
            }
        });

        /* Route Performance Heatmap Table */
        if (D.route_performance && D.route_performance.length) {
            const tblPerf = document.getElementById("tbl-route-perf");
            if (tblPerf) {
                tblPerf.innerHTML = `<thead><tr><th>Route</th><th>Revenue</th><th>Load Factor</th><th>Yield</th><th>OTP</th><th>RASM</th></tr></thead><tbody>` +
                    D.route_performance.map(r => {
                        const lfCls = r.lf > 0.85 ? "text-success" : r.lf < 0.7 ? "text-danger" : "text-warning";
                        const otpCls = r.otp > 0.9 ? "text-success" : r.otp < 0.75 ? "text-danger" : "text-warning";
                        return `<tr><td class="fw-bold">${r.route}</td><td>Rs.${INR(r.rev)}</td><td class="${lfCls} fw-bold">${PCT(r.lf)}</td><td>Rs.${r.yld.toFixed(2)}</td><td class="${otpCls} fw-bold">${PCT(r.otp)}</td><td>${r.rasm_val.toFixed(4)}</td></tr>`;
                    }).join("") + `</tbody>`;
            }
        }

        /* O-D Flows Table */
        if (D.route_flows && D.route_flows.length) {
            const tblOD = document.getElementById("tbl-od-flows");
            if (tblOD) {
                tblOD.innerHTML = `<thead><tr><th>Origin</th><th>Dest</th><th>Passengers</th><th>Revenue</th></tr></thead><tbody>` +
                    D.route_flows.map(r => `<tr><td class="fw-bold">${r.origin_airport}</td><td class="fw-bold">${r.destination_airport}</td><td>${INR(r.pax)}</td><td>Rs.${INR(r.revenue)}</td></tr>`).join("") + `</tbody>`;
            }
        }
    }

    /* =============================================================
       TAB 3 – BOOKING INTELLIGENCE
       ============================================================= */
    function renderBookings(D) {
        /* Lead time histogram vertical bar */
        make("c-lead", {
            type: "bar",
            data: {
                labels: D.lead_time_dist.map(d => d.bucket),
                datasets: [{
                    label: "Bookings",
                    data: D.lead_time_dist.map(d => d.count),
                    backgroundColor: alpha(COLORS.indigo, 0.7),
                    borderRadius: 6, maxBarThickness: 50,
                }]
            },
            options: {
                responsive: true, maintainAspectRatio: false,
                plugins: { legend: { display: false } },
                scales: { x: { grid: { display: false } }, y: { beginAtZero: true } }
            }
        });

        /* Channel vertical bar */
        make("c-channel", {
            type: "bar",
            data: {
                labels: D.channel_dist.map(d => d.channel),
                datasets: [{
                    label: "Bookings",
                    data: D.channel_dist.map(d => d.count),
                    backgroundColor: [alpha(COLORS.blue, 0.7), alpha(COLORS.green, 0.7), alpha(COLORS.amber, 0.7), alpha(COLORS.purple, 0.7)],
                    borderRadius: 6, maxBarThickness: 60,
                }]
            },
            options: {
                responsive: true, maintainAspectRatio: false,
                plugins: { legend: { display: false } },
                scales: { x: { grid: { display: false }, ticks: { font: { size: 12, weight: "600" } } }, y: { beginAtZero: true } }
            }
        });

        /* DOW vertical bar */
        make("c-dow", {
            type: "bar",
            data: {
                labels: D.booking_dow.map(d => d.day),
                datasets: [{
                    label: "Bookings",
                    data: D.booking_dow.map(d => d.count),
                    backgroundColor: PALETTE.slice(0, 7).map(c => alpha(c, 0.7)),
                    borderRadius: 6, maxBarThickness: 50,
                }]
            },
            options: {
                responsive: true, maintainAspectRatio: false,
                plugins: { legend: { display: false } },
                scales: { x: { grid: { display: false } }, y: { beginAtZero: true } }
            }
        });

        /* TOD vertical bar */
        make("c-tod", {
            type: "bar",
            data: {
                labels: D.booking_tod.map(d => d.period),
                datasets: [{
                    label: "Bookings",
                    data: D.booking_tod.map(d => d.count),
                    backgroundColor: [alpha(COLORS.amber, 0.7), alpha(COLORS.blue, 0.7), alpha(COLORS.purple, 0.7), alpha(COLORS.indigo, 0.7)],
                    borderRadius: 6, maxBarThickness: 60,
                }]
            },
            options: {
                responsive: true, maintainAspectRatio: false,
                plugins: { legend: { display: false } },
                scales: { x: { grid: { display: false } }, y: { beginAtZero: true } }
            }
        });

        /* Pace tables */
        function paceTable(id, rows, good) {
            const el = document.getElementById(id);
            if (!el || !rows.length) return;
            el.innerHTML = `<thead><tr><th>Flight</th><th>Route</th><th>Pace</th><th>Confirmed</th></tr></thead><tbody>` +
                rows.map(r => {
                    const cls = good ? "text-success" : "text-danger";
                    return `<tr><td>${r.flight_id}</td><td>${r.route}</td><td class="${cls} fw-bold">${r.pace_vs_historical}x</td><td>${r.total_confirmed}</td></tr>`;
                }).join("") + `</tbody>`;
        }
        paceTable("tbl-behind", D.pace.behind, false);
        paceTable("tbl-ahead", D.pace.ahead, true);

        /* Booking Pace Deviation by Route */
        if (D.mape_by_route && D.mape_by_route.length) {
            make("c-mape", {
                type: "bar",
                data: {
                    labels: D.mape_by_route.map(d => d.route),
                    datasets: [{
                        label: "Pace Deviation %",
                        data: D.mape_by_route.map(d => d.mape_pct),
                        backgroundColor: D.mape_by_route.map(d =>
                            d.mape_pct > 15 ? alpha(COLORS.red, 0.75) : d.mape_pct > 8 ? alpha(COLORS.amber, 0.75) : alpha(COLORS.green, 0.75)
                        ),
                        borderRadius: 4, maxBarThickness: 40,
                    }]
                },
                options: {
                    responsive: true, maintainAspectRatio: false,
                    plugins: { legend: { display: false } },
                    scales: {
                        x: { ticks: { font: { size: 10 }, maxRotation: 45, minRotation: 45 }, grid: { display: false } },
                        y: { ticks: { callback: v => v + "%" }, beginAtZero: true }
                    }
                }
            });
        }
    }

    /* =============================================================
       TAB 4 – PRICING & COMPETITORS
       ============================================================= */
    function renderPricing(D) {
        /* Price position vertical bar */
        if (D.price_position && D.price_position.length) {
            make("c-price-pos", {
                type: "bar",
                data: {
                    labels: D.price_position.map(d => d.position),
                    datasets: [{
                        label: "Count",
                        data: D.price_position.map(d => d.count),
                        backgroundColor: [alpha(COLORS.green, 0.7), alpha(COLORS.amber, 0.7), alpha(COLORS.red, 0.7), alpha(COLORS.purple, 0.7)],
                        borderRadius: 6, maxBarThickness: 60,
                    }]
                },
                options: {
                    responsive: true, maintainAspectRatio: false,
                    plugins: { legend: { display: false } },
                    scales: { x: { grid: { display: false }, ticks: { font: { size: 12, weight: "600" } } }, y: { beginAtZero: true } }
                }
            });
        }

        /* Market share by route (vertical) */
        if (D.competitive_routes && D.competitive_routes.length) {
            make("c-comp-route", {
                type: "bar",
                data: {
                    labels: D.competitive_routes.map(d => d.route),
                    datasets: [{
                        label: "Market Share",
                        data: D.competitive_routes.map(d => d.market_share * 100),
                        backgroundColor: alpha(COLORS.blue, 0.7),
                        borderRadius: 4, maxBarThickness: 40,
                    }]
                },
                options: {
                    responsive: true, maintainAspectRatio: false,
                    plugins: { legend: { display: false } },
                    scales: {
                        x: { ticks: { font: { size: 10 }, maxRotation: 45, minRotation: 45 }, grid: { display: false } },
                        y: { min: 0, max: 100, ticks: { callback: v => v + "%" }, beginAtZero: true }
                    }
                }
            });

            /* Competition table */
            const tbl = document.getElementById("tbl-comp");
            if (tbl) {
                tbl.innerHTML = `<thead><tr><th>Route</th><th>Competitors</th><th>Market Share</th><th>Price Index</th></tr></thead><tbody>` +
                    D.competitive_routes.map(r => {
                        const cls = r.price_idx > 1.05 ? "text-danger" : (r.price_idx < 0.95 ? "text-success" : "");
                        return `<tr><td>${r.route}</td><td>${r.competitors}</td><td>${PCT(r.market_share)}</td><td class="${cls} fw-bold">${r.price_idx.toFixed(3)}</td></tr>`;
                    }).join("") + `</tbody>`;
            }
        }

        /* Price Elasticity – Revenue Impact */
        if (D.price_elasticity && D.price_elasticity.length) {
            make("c-elasticity", {
                type: "bar",
                data: {
                    labels: D.price_elasticity.map(d => (d.price_change >= 0 ? "+" : "") + d.price_change + "%"),
                    datasets: [{
                        label: "Revenue Impact (Rs.)",
                        data: D.price_elasticity.map(d => d.rev_impact),
                        backgroundColor: D.price_elasticity.map(d =>
                            d.rev_impact >= 0 ? alpha(COLORS.green, 0.75) : alpha(COLORS.red, 0.75)
                        ),
                        borderRadius: 4, maxBarThickness: 40,
                    }]
                },
                options: {
                    responsive: true, maintainAspectRatio: false,
                    plugins: { legend: { display: false },
                        tooltip: { callbacks: { label: ctx => "Rs." + INR(ctx.raw) } }
                    },
                    scales: {
                        x: { grid: { display: false }, title: { display: true, text: "Price Change", font: { size: 11 } } },
                        y: { ticks: { callback: v => "Rs." + INR(v) }, title: { display: true, text: "Revenue Impact", font: { size: 11 } } }
                    }
                }
            });

            /* Estimated Passengers vs Price Change */
            make("c-elasticity-pax", {
                type: "line",
                data: {
                    labels: D.price_elasticity.map(d => (d.price_change >= 0 ? "+" : "") + d.price_change + "%"),
                    datasets: [{
                        label: "Est. Passengers",
                        data: D.price_elasticity.map(d => d.est_pax),
                        borderColor: COLORS.blue,
                        backgroundColor: alpha(COLORS.blue, 0.15),
                        fill: true, tension: 0.3, pointRadius: 5, pointStyle: "circle", borderWidth: 2,
                    }]
                },
                options: {
                    responsive: true, maintainAspectRatio: false,
                    plugins: { legend: { display: false } },
                    scales: {
                        x: { grid: { display: false }, title: { display: true, text: "Price Change", font: { size: 11 } } },
                        y: { title: { display: true, text: "Passengers", font: { size: 11 } }, ticks: { callback: v => INR(v) } }
                    }
                }
            });
        }
    }

    /* =============================================================
       TAB 5 – OVERBOOKING & NO-SHOWS
       ============================================================= */
    function renderOverbooking(D) {
        const ob = D.overbooking;
        document.getElementById("ob-qty").textContent = ob.avg_overbook_qty;
        document.getElementById("ob-denied").textContent = "Rs." + INR(ob.denied_boarding_cost);
        document.getElementById("ob-empty").textContent = "Rs." + INR(ob.avg_empty_seat_cost);
        document.getElementById("ob-bump").textContent = "Rs." + INR(ob.voluntary_bump);

        /* Status counts */
        const k = D.kpis;
        document.getElementById("st-confirmed").textContent = INR(k.confirmed);
        document.getElementById("st-cancelled").textContent = INR(k.cancelled);
        document.getElementById("st-noshow").textContent = INR(k.noshows);

        /* Status vertical bar */
        make("c-status", {
            type: "bar",
            data: {
                labels: ["Confirmed", "Cancelled", "No-Show"],
                datasets: [{
                    label: "Count",
                    data: [k.confirmed, k.cancelled, k.noshows],
                    backgroundColor: [alpha(COLORS.green, 0.7), alpha(COLORS.red, 0.7), alpha(COLORS.purple, 0.7)],
                    borderRadius: 6, maxBarThickness: 60,
                }]
            },
            options: {
                responsive: true, maintainAspectRatio: false,
                plugins: { legend: { display: false } },
                scales: { x: { grid: { display: false }, ticks: { font: { size: 12, weight: "600" } } }, y: { beginAtZero: true } }
            }
        });

        /* No-show by route (vertical) */
        if (D.noshow_by_route && D.noshow_by_route.length) {
            make("c-noshow", {
                type: "bar",
                data: {
                    labels: D.noshow_by_route.map(d => d.route),
                    datasets: [{
                        label: "No-Show Rate",
                        data: D.noshow_by_route.map(d => d.noshow_rate * 100),
                        backgroundColor: alpha(COLORS.red, 0.7),
                        borderRadius: 4, maxBarThickness: 35,
                    }]
                },
                options: {
                    responsive: true, maintainAspectRatio: false,
                    plugins: { legend: { display: false } },
                    scales: {
                        x: { ticks: { font: { size: 10 }, maxRotation: 45, minRotation: 45 }, grid: { display: false } },
                        y: { ticks: { callback: v => v.toFixed(1) + "%" }, beginAtZero: true }
                    }
                }
            });
        }

        /* Cancel by horizon vertical bar */
        if (D.cancel_by_horizon && D.cancel_by_horizon.length) {
            make("c-cancel", {
                type: "bar",
                data: {
                    labels: D.cancel_by_horizon.map(d => d.bucket),
                    datasets: [{
                        label: "Cancel Rate",
                        data: D.cancel_by_horizon.map(d => d.rate * 100),
                        backgroundColor: alpha(COLORS.amber, 0.7),
                        borderRadius: 6, maxBarThickness: 50,
                    }]
                },
                options: {
                    responsive: true, maintainAspectRatio: false,
                    plugins: { legend: { display: false } },
                    scales: {
                        x: { grid: { display: false } },
                        y: { ticks: { callback: v => v.toFixed(1) + "%" }, beginAtZero: true }
                    }
                }
            });
        }
    }

    /* =============================================================
       TAB 6 – OPERATIONS
       ============================================================= */
    function renderOperations(D) {
        /* OTP by route (vertical) */
        if (D.otp_by_route && D.otp_by_route.length) {
            make("c-otp", {
                type: "bar",
                data: {
                    labels: D.otp_by_route.map(d => d.route),
                    datasets: [
                        {
                            label: "On-Time %",
                            data: D.otp_by_route.map(d => d.otp * 100),
                            backgroundColor: alpha(COLORS.green, 0.7),
                            borderRadius: 4, maxBarThickness: 35,
                            yAxisID: "y", order: 2,
                        },
                        {
                            label: "Avg Delay (min)",
                            data: D.otp_by_route.map(d => d.delay),
                            type: "line",
                            borderColor: COLORS.red,
                            backgroundColor: COLORS.red,
                            pointRadius: 4, pointStyle: "triangle",
                            borderWidth: 2,
                            yAxisID: "y1", order: 1,
                        }
                    ]
                },
                options: {
                    responsive: true, maintainAspectRatio: false,
                    scales: {
                        x: { ticks: { font: { size: 10 }, maxRotation: 45, minRotation: 45 }, grid: { display: false } },
                        y: { ticks: { callback: v => v + "%" }, beginAtZero: true },
                        y1: { position: "right", grid: { drawOnChartArea: false }, title: { display: true, text: "Delay (min)", font: { size: 10 } } }
                    }
                }
            });
        }

        /* Congestion vertical bar */
        if (D.congestion && D.congestion.length) {
            make("c-congestion", {
                type: "bar",
                data: {
                    labels: D.congestion.map(d => d.airport),
                    datasets: [{
                        label: "Congestion Index",
                        data: D.congestion.map(d => d.congestion),
                        backgroundColor: D.congestion.map(d =>
                            d.congestion > 0.9 ? alpha(COLORS.red, 0.75) : (d.congestion > 0.7 ? alpha(COLORS.amber, 0.75) : alpha(COLORS.green, 0.75))
                        ),
                        borderRadius: 4, maxBarThickness: 40,
                    }]
                },
                options: {
                    responsive: true, maintainAspectRatio: false,
                    plugins: { legend: { display: false } },
                    scales: {
                        x: { ticks: { font: { size: 10 }, maxRotation: 45, minRotation: 45 }, grid: { display: false } },
                        y: { min: 0, max: 1.1, beginAtZero: true }
                    }
                }
            });
        }

        /* Crew Availability by Route */
        if (D.crew_by_route && D.crew_by_route.length) {
            make("c-crew", {
                type: "bar",
                data: {
                    labels: D.crew_by_route.map(d => d.route),
                    datasets: [{
                        label: "Crew Availability Index",
                        data: D.crew_by_route.map(d => d.crew_avail),
                        backgroundColor: D.crew_by_route.map(d =>
                            d.crew_avail < 0.7 ? alpha(COLORS.red, 0.75) : d.crew_avail < 0.85 ? alpha(COLORS.amber, 0.75) : alpha(COLORS.green, 0.75)
                        ),
                        borderRadius: 4, maxBarThickness: 35,
                    }]
                },
                options: {
                    responsive: true, maintainAspectRatio: false,
                    plugins: { legend: { display: false } },
                    scales: {
                        x: { ticks: { font: { size: 10 }, maxRotation: 45, minRotation: 45 }, grid: { display: false } },
                        y: { min: 0, max: 1.1, beginAtZero: true, title: { display: true, text: "Availability Index", font: { size: 10 } } }
                    }
                }
            });

            /* Maintenance Rate by Route */
            if (D.crew_by_route[0].maint_rate !== undefined) {
                make("c-maint", {
                    type: "bar",
                    data: {
                        labels: D.crew_by_route.map(d => d.route),
                        datasets: [{
                            label: "Maintenance Rate",
                            data: D.crew_by_route.map(d => d.maint_rate),
                            backgroundColor: D.crew_by_route.map(d =>
                                d.maint_rate > 0.15 ? alpha(COLORS.red, 0.75) : d.maint_rate > 0.08 ? alpha(COLORS.amber, 0.75) : alpha(COLORS.green, 0.75)
                            ),
                            borderRadius: 4, maxBarThickness: 35,
                        }]
                    },
                    options: {
                        responsive: true, maintainAspectRatio: false,
                        plugins: { legend: { display: false } },
                        scales: {
                            x: { ticks: { font: { size: 10 }, maxRotation: 45, minRotation: 45 }, grid: { display: false } },
                            y: { beginAtZero: true, ticks: { callback: v => (v * 100).toFixed(0) + "%" }, title: { display: true, text: "Maint. Rate", font: { size: 10 } } }
                        }
                    }
                });
            }
        }
    }

    /* =============================================================
       TAB 7 – CUSTOMER SEGMENTS
       ============================================================= */
    function renderSegments(D) {
        const seg = D.segmentation;

        /* Business vs Leisure vertical bar */
        make("c-biz", {
            type: "bar",
            data: {
                labels: ["Business", "Leisure"],
                datasets: [{
                    label: "Count",
                    data: [seg.business, seg.leisure],
                    backgroundColor: [alpha(COLORS.blue, 0.7), alpha(COLORS.amber, 0.7)],
                    borderRadius: 6, maxBarThickness: 60,
                }]
            },
            options: {
                responsive: true, maintainAspectRatio: false,
                plugins: { legend: { display: false } },
                scales: { x: { grid: { display: false }, ticks: { font: { size: 13, weight: "600" } } }, y: { beginAtZero: true } }
            }
        });

        /* Loyalty tier vertical bar */
        if (seg.loyalty && seg.loyalty.length) {
            make("c-loyalty", {
                type: "bar",
                data: {
                    labels: seg.loyalty.map(d => d.tier),
                    datasets: [{
                        label: "Count",
                        data: seg.loyalty.map(d => d.count),
                        backgroundColor: PALETTE.slice(0, seg.loyalty.length).map(c => alpha(c, 0.7)),
                        borderRadius: 6, maxBarThickness: 60,
                    }]
                },
                options: {
                    responsive: true, maintainAspectRatio: false,
                    plugins: { legend: { display: false } },
                    scales: { x: { grid: { display: false }, ticks: { font: { size: 12, weight: "600" } } }, y: { beginAtZero: true } }
                }
            });
        }

        /* Fare class vertical bar */
        if (seg.fare_class && seg.fare_class.length) {
            make("c-fare", {
                type: "bar",
                data: {
                    labels: seg.fare_class.map(d => d.fare_class),
                    datasets: [{
                        label: "Count",
                        data: seg.fare_class.map(d => d.count),
                        backgroundColor: [alpha(COLORS.blue, 0.7), alpha(COLORS.amber, 0.7), alpha(COLORS.purple, 0.7)],
                        borderRadius: 6, maxBarThickness: 60,
                    }]
                },
                options: {
                    responsive: true, maintainAspectRatio: false,
                    plugins: { legend: { display: false } },
                    scales: { x: { grid: { display: false }, ticks: { font: { size: 13, weight: "600" } } }, y: { beginAtZero: true } }
                }
            });
        }

        /* Customer Lifetime Value Distribution */
        if (D.clv_distribution && D.clv_distribution.length) {
            make("c-clv", {
                type: "bar",
                data: {
                    labels: D.clv_distribution.map(d => d.bucket),
                    datasets: [{
                        label: "Customers",
                        data: D.clv_distribution.map(d => d.count),
                        backgroundColor: [
                            alpha(COLORS.red, 0.7), alpha(COLORS.amber, 0.7), alpha(COLORS.lime, 0.7),
                            alpha(COLORS.green, 0.7), alpha(COLORS.blue, 0.7), alpha(COLORS.purple, 0.7),
                        ],
                        borderRadius: 6, maxBarThickness: 55,
                    }]
                },
                options: {
                    responsive: true, maintainAspectRatio: false,
                    plugins: { legend: { display: false } },
                    scales: {
                        x: { grid: { display: false }, title: { display: true, text: "Lifetime Spend Bucket", font: { size: 10 } } },
                        y: { beginAtZero: true, title: { display: true, text: "Customers", font: { size: 10 } } }
                    }
                }
            });
        }

        /* CLV by Loyalty Tier */
        if (D.clv_by_tier && D.clv_by_tier.length) {
            make("c-clv-tier", {
                type: "bar",
                data: {
                    labels: D.clv_by_tier.map(d => d.loyalty_tier),
                    datasets: [
                        {
                            label: "Avg CLV (Rs.)",
                            data: D.clv_by_tier.map(d => d.avg_clv),
                            backgroundColor: alpha(COLORS.indigo, 0.7),
                            borderRadius: 4, maxBarThickness: 40,
                            yAxisID: "y",
                        },
                        {
                            label: "Customers",
                            data: D.clv_by_tier.map(d => d.customers),
                            type: "line",
                            borderColor: COLORS.amber,
                            backgroundColor: COLORS.amber,
                            pointRadius: 5, pointStyle: "circle", borderWidth: 2,
                            yAxisID: "y1",
                        }
                    ]
                },
                options: {
                    responsive: true, maintainAspectRatio: false,
                    scales: {
                        x: { grid: { display: false }, ticks: { font: { size: 12, weight: "600" } } },
                        y: { ticks: { callback: v => "Rs." + INR(v) }, beginAtZero: true, title: { display: true, text: "Avg CLV", font: { size: 10 } } },
                        y1: { position: "right", grid: { drawOnChartArea: false }, beginAtZero: true, title: { display: true, text: "Customers", font: { size: 10 } } }
                    }
                }
            });
        }
    }

    /* =============================================================
       TAB 9 – ANCILLARY REVENUE
       ============================================================= */
    function renderAncillary(D) {
        /* Ancillary Breakdown by Category */
        if (D.ancillary_breakdown && D.ancillary_breakdown.length) {
            make("c-anc-cat", {
                type: "bar",
                data: {
                    labels: D.ancillary_breakdown.map(d => d.category),
                    datasets: [{
                        label: "Revenue (Rs.)",
                        data: D.ancillary_breakdown.map(d => d.revenue),
                        backgroundColor: PALETTE.slice(0, 6).map(c => alpha(c, 0.75)),
                        borderRadius: 6, maxBarThickness: 50,
                    }]
                },
                options: {
                    responsive: true, maintainAspectRatio: false,
                    plugins: { legend: { display: false } },
                    scales: {
                        x: { grid: { display: false }, ticks: { font: { size: 10, weight: "600" }, maxRotation: 30 } },
                        y: { ticks: { callback: v => "Rs." + INR(v) }, beginAtZero: true }
                    }
                }
            });

            /* Detail table */
            const tblAnc = document.getElementById("tbl-anc-detail");
            if (tblAnc) {
                const totalRev = D.ancillary_breakdown.reduce((s, d) => s + d.revenue, 0);
                tblAnc.innerHTML = `<thead><tr><th>Service</th><th>Revenue</th><th>Share</th><th>Contribution</th></tr></thead><tbody>` +
                    D.ancillary_breakdown.map(d => {
                        const barW = d.pct;
                        return `<tr><td class="fw-bold">${d.category}</td><td>Rs.${INR(d.revenue)}</td><td>${d.pct}%</td>
                            <td><div class="progress" style="height:8px"><div class="progress-bar bg-primary" style="width:${barW}%"></div></div></td></tr>`;
                    }).join("") +
                    `<tr class="table-dark fw-bold"><td>Total</td><td>Rs.${INR(totalRev)}</td><td>100%</td><td></td></tr></tbody>`;
            }
        }

        /* Ancillary by Route */
        if (D.ancillary_by_route && D.ancillary_by_route.length) {
            make("c-anc-route", {
                type: "bar",
                data: {
                    labels: D.ancillary_by_route.map(d => d.route),
                    datasets: [{
                        label: "Avg Ancillary/Pax",
                        data: D.ancillary_by_route.map(d => d.avg_ancillary),
                        backgroundColor: alpha(COLORS.teal, 0.7),
                        borderRadius: 4, maxBarThickness: 40,
                    }]
                },
                options: {
                    responsive: true, maintainAspectRatio: false,
                    plugins: { legend: { display: false } },
                    scales: {
                        x: { ticks: { font: { size: 10 }, maxRotation: 45, minRotation: 45 }, grid: { display: false } },
                        y: { ticks: { callback: v => "Rs." + v }, beginAtZero: true }
                    }
                }
            });
        }

        /* Ancillary by Fare Class */
        if (D.ancillary_by_class && D.ancillary_by_class.length) {
            make("c-anc-class", {
                type: "bar",
                data: {
                    labels: D.ancillary_by_class.map(d => d.fare_class),
                    datasets: [
                        {
                            label: "Avg Ancillary/Pax",
                            data: D.ancillary_by_class.map(d => d.avg_anc),
                            backgroundColor: alpha(COLORS.purple, 0.7),
                            borderRadius: 6, maxBarThickness: 55,
                            yAxisID: "y",
                        },
                        {
                            label: "Total Revenue",
                            data: D.ancillary_by_class.map(d => d.total_anc),
                            type: "line",
                            borderColor: COLORS.green,
                            backgroundColor: COLORS.green,
                            pointRadius: 6, pointStyle: "rectRot", borderWidth: 2,
                            yAxisID: "y1",
                        }
                    ]
                },
                options: {
                    responsive: true, maintainAspectRatio: false,
                    scales: {
                        x: { grid: { display: false }, ticks: { font: { size: 12, weight: "600" } } },
                        y: { ticks: { callback: v => "Rs." + v }, beginAtZero: true, title: { display: true, text: "Avg/Pax", font: { size: 10 } } },
                        y1: { position: "right", grid: { drawOnChartArea: false }, ticks: { callback: v => "Rs." + INR(v) }, title: { display: true, text: "Total", font: { size: 10 } } }
                    }
                }
            });
        }
    }

    /* =============================================================
       TAB 10 – FORECAST & CAPACITY
       ============================================================= */
    function renderForecast(D) {
        /* Revenue vs Target */
        const revTarget = D.revenue_target || D.rev_target;
        if (revTarget && revTarget.length) {
            const months = ["", "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
            const rtLabels = revTarget.map(d => d.label || (months[d.month] || d.month));
            make("c-rev-target", {
                type: "bar",
                data: {
                    labels: rtLabels,
                    datasets: [
                        { label: "Actual Revenue", data: revTarget.map(d => d.actual), backgroundColor: alpha(COLORS.blue, 0.7), borderRadius: 4, maxBarThickness: 35, order: 2 },
                        { label: "Target (10% Growth)", data: revTarget.map(d => d.target), type: "line", borderColor: COLORS.red, pointRadius: 5, borderWidth: 2, borderDash: [6, 3], order: 1, fill: false }
                    ]
                },
                options: {
                    responsive: true, maintainAspectRatio: false,
                    plugins: {
                        title: { display: true, text: "Jan 2025 – Feb 2026", font: { size: 12 }, color: "#64748b", padding: { bottom: 10 } }
                    },
                    scales: {
                        x: { grid: { display: false }, ticks: { font: { size: 10 }, maxRotation: 45 } },
                        y: { ticks: { callback: v => "Rs." + INR(v) }, beginAtZero: true }
                    }
                }
            });
        }

        /* Competitor Capacity */
        const compCap = D.competitor_capacity || D.comp_capacity;
        if (compCap && compCap.length) {
            make("c-comp-cap", {
                type: "bar",
                data: {
                    labels: compCap.map(d => d.route),
                    datasets: [
                        { label: "Our Seats", data: compCap.map(d => d.our_seats), backgroundColor: alpha(COLORS.blue, 0.7), borderRadius: 4, maxBarThickness: 30 },
                        { label: "Competitor Seats", data: compCap.map(d => d.comp_seats), backgroundColor: alpha(COLORS.red, 0.5), borderRadius: 4, maxBarThickness: 30 }
                    ]
                },
                options: {
                    responsive: true, maintainAspectRatio: false,
                    scales: {
                        x: { ticks: { font: { size: 9 }, maxRotation: 45 }, grid: { display: false } },
                        y: { beginAtZero: true }
                    }
                }
            });
        }

        /* 30/60/90 Day Demand Forecast with CI */
        if (D.demand_forecast && D.demand_forecast.length) {
            /* Data is route-based: {route, 30d:{demand,lower,upper}, 60d:..., 90d:...} */
            const dfRoutes = D.demand_forecast.map(d => d.route);
            make("c-demand-forecast", {
                type: "bar",
                data: {
                    labels: dfRoutes,
                    datasets: [
                        { label: "30-Day Forecast", data: D.demand_forecast.map(d => d["30d"]?.demand || 0), backgroundColor: alpha(COLORS.blue, 0.7), borderRadius: 6, maxBarThickness: 40 },
                        { label: "60-Day Forecast", data: D.demand_forecast.map(d => d["60d"]?.demand || 0), backgroundColor: alpha(COLORS.cyan, 0.7), borderRadius: 6, maxBarThickness: 40 },
                        { label: "90-Day Forecast", data: D.demand_forecast.map(d => d["90d"]?.demand || 0), backgroundColor: alpha(COLORS.purple, 0.7), borderRadius: 6, maxBarThickness: 40 }
                    ]
                },
                options: {
                    responsive: true, maintainAspectRatio: false,
                    scales: {
                        x: { grid: { display: false }, ticks: { font: { size: 12, weight: "600" } } },
                        y: { beginAtZero: true, title: { display: true, text: "Passengers", font: { size: 10 } } }
                    }
                }
            });
        }

        /* Capacity vs Demand by Route */
        const capDem = D.capacity_vs_demand || D.cap_vs_demand;
        if (capDem && capDem.length) {
            make("c-cap-demand", {
                type: "bar",
                data: {
                    labels: capDem.map(d => d.route),
                    datasets: [
                        { label: "Capacity", data: capDem.map(d => d.total_capacity || d.capacity), backgroundColor: alpha(COLORS.cyan, 0.6), borderRadius: 4, maxBarThickness: 25 },
                        { label: "Demand", data: capDem.map(d => d.total_booked || d.demand), backgroundColor: alpha(COLORS.amber, 0.7), borderRadius: 4, maxBarThickness: 25 }
                    ]
                },
                options: {
                    responsive: true, maintainAspectRatio: false,
                    indexAxis: "y",
                    scales: {
                        x: { beginAtZero: true },
                        y: { ticks: { font: { size: 10 } }, grid: { display: false } }
                    }
                }
            });
        }
    }

    /* =============================================================
       TAB 11 – ALERTS
       ============================================================= */
    function renderAlerts(D) {
        if (!D.alerts || !D.alerts.length) {
            document.getElementById("alerts-list").innerHTML = '<div class="text-muted text-center py-4">No alerts at this time</div>';
            return;
        }
        const alerts = D.alerts;
        const critical = alerts.filter(a => a.severity === "high" && (a.type === "danger" || a.type === "warning")).length;
        const warnings = alerts.filter(a => a.type === "warning").length;
        const info = alerts.filter(a => a.type === "info").length;
        const positive = alerts.filter(a => a.type === "success").length;

        document.getElementById("alert-critical").textContent = critical;
        document.getElementById("alert-warnings").textContent = warnings;
        document.getElementById("alert-info").textContent = info;
        document.getElementById("alert-positive").textContent = positive;

        const badge = document.getElementById("alert-count-badge");
        if (badge && alerts.length > 0) { badge.textContent = alerts.length; badge.style.display = "inline"; }

        const iconMap = { danger: "bi-exclamation-triangle-fill", warning: "bi-exclamation-circle-fill", info: "bi-info-circle-fill", success: "bi-check-circle-fill" };
        document.getElementById("alerts-list").innerHTML = alerts.map(a => {
            const icon = iconMap[a.type] || "bi-bell-fill";
            return `<div class="alert alert-${a.type} d-flex align-items-start py-2 px-3 mb-2" style="border-radius:8px;">
                <i class="bi ${icon} me-2 mt-1"></i>
                <div class="flex-grow-1">
                    <span class="badge bg-${a.type} me-1">${a.category}</span>
                    <span class="small">${a.message}</span>
                </div>
                <span class="badge bg-dark">${a.severity}</span>
            </div>`;
        }).join("");
    }

    /* =============================================================
       TAB 12 – 365-DAY PRICING FORECAST
       ============================================================= */
    let _365data = null;
    let _365trendChart = null;

    function setup365PricingTab() {
        const btn = document.getElementById("btn-load-365");
        if (!btn) return;

        /* Show/hide custom distance input */
        const routeSel = document.getElementById("sel-365-route");
        const distWrap = document.getElementById("custom-dist-wrap");
        if (routeSel && distWrap) {
            routeSel.addEventListener("change", () => {
                distWrap.style.display = routeSel.value === "CUSTOM" ? "" : "none";
            });
        }

        btn.addEventListener("click", async () => {
            btn.disabled = true;
            btn.innerHTML = '<span class="spinner-border spinner-border-sm me-1"></span>Generating…';
            try {
                /* Build URL with route or custom distance */
                let url = "/api/pricing/365";
                const route = routeSel ? routeSel.value : "PNQ-NAG";
                if (route === "CUSTOM") {
                    const dist = document.getElementById("inp-365-dist");
                    url += `?route=CUSTOM&distance_km=${dist ? dist.value : 600}`;
                } else {
                    url += `?route=${route}`;
                }
                const res = await fetch(url);
                if (!res.ok) throw new Error("API error " + res.status);
                _365data = await res.json();
                render365Dashboard(_365data);
                btn.innerHTML = '<i class="bi bi-arrow-clockwise me-1"></i>Refresh';
            } catch (e) {
                btn.closest(".tab-pane").insertAdjacentHTML("afterbegin",
                    `<div class="alert alert-danger m-2">${e.message}</div>`);
            } finally {
                btn.disabled = false;
            }
        });
        /* Segment filter */
        const sel = document.getElementById("sel-365-segment");
        if (sel) sel.addEventListener("change", () => {
            if (_365data) render365Trend(_365data, sel.value);
        });
    }

    function render365Dashboard(D) {
        const s = D.summary;
        const ei = (id, v) => { const e = document.getElementById(id); if (e) e.textContent = v; };
        ei("p365-route", D.route || "—");
        ei("p365-dist", (D.distance_km || 0) + " km");
        ei("p365-base", "\u20B9" + INR(D.base_price || 0));
        ei("p365-range", "\u20B9" + INR(D.price_floor || 0) + " – \u20B9" + INR(D.price_ceiling || 0));
        ei("p365-avg", "\u20B9" + INR(s.avg_price));
        ei("p365-min", "\u20B9" + INR(s.min_price));
        ei("p365-min-date", s.min_date);
        ei("p365-max", "\u20B9" + INR(s.max_price));
        ei("p365-max-date", s.max_date);
        ei("p365-events", s.event_days);
        ei("p365-wedding", (s.wedding_season_days || 0) + "d");
        ei("p365-school", (s.school_break_days || 0) + "d");

        render365Trend(D, "all");
        render365Monthly(D);
        render365DoW(D);
        render365EventTable(D);
    }

    function render365Trend(D, segment) {
        const daily = D.daily;
        const labels = daily.map(d => d.date);
        let prices;
        if (segment === "all") {
            prices = daily.map(d => d.avg_price);
        } else {
            prices = daily.map(d => d.prices[segment]);
        }
        /* Color code: green when below avg, red when above */
        const avg = prices.reduce((a, b) => a + b, 0) / prices.length;

        /* Event markers — find indices of event days */
        const eventIndices = daily.map((d, i) => d.event !== "none" ? i : -1).filter(i => i >= 0);
        const eventPrices = eventIndices.map(i => prices[i]);
        const eventLabels = eventIndices.map(i => labels[i]);

        const canvas = document.getElementById("c-365-trend");
        if (!canvas) return;
        if (_365trendChart) _365trendChart.destroy();

        _365trendChart = new Chart(canvas, {
            type: "line",
            data: {
                labels: labels,
                datasets: [
                    {
                        label: segment === "all" ? "Avg Price" : segment.charAt(0).toUpperCase() + segment.slice(1) + " Price",
                        data: prices,
                        borderColor: COLORS.purple,
                        backgroundColor: alpha(COLORS.purple, 0.08),
                        borderWidth: 1.5,
                        pointRadius: 0,
                        pointHitRadius: 6,
                        fill: true,
                        tension: 0.3,
                    },
                    {
                        label: "Avg Baseline",
                        data: Array(365).fill(avg),
                        borderColor: alpha(COLORS.blue, 0.4),
                        borderWidth: 1,
                        borderDash: [5, 5],
                        pointRadius: 0,
                        fill: false,
                    },
                    {
                        label: "Events / Festivals",
                        data: labels.map((_, i) => eventIndices.includes(i) ? prices[i] : null),
                        type: "scatter",
                        pointRadius: 5,
                        pointStyle: "triangle",
                        pointBackgroundColor: COLORS.amber,
                        pointBorderColor: COLORS.amber,
                        showLine: false,
                    },
                ]
            },
            options: {
                responsive: true, maintainAspectRatio: false,
                interaction: { mode: "index", intersect: false },
                scales: {
                    x: {
                        type: "category",
                        ticks: {
                            maxTicksLimit: 12,
                            callback: function(v, i) {
                                const dt = labels[i];
                                if (!dt) return "";
                                const m = new Date(dt);
                                return m.toLocaleDateString("en-IN", { month: "short", day: "numeric" });
                            },
                            font: { size: 10 },
                        },
                        grid: { display: false },
                    },
                    y: {
                        ticks: { callback: v => "\u20B9" + INR(v) },
                        beginAtZero: false,
                        suggestedMin: (D.price_floor || Math.min(...prices)) * 0.9,
                        suggestedMax: (D.price_ceiling || Math.max(...prices)) * 1.05,
                    },
                },
                plugins: {
                    tooltip: {
                        callbacks: {
                            title: ctx => {
                                const i = ctx[0].dataIndex;
                                const d = daily[i];
                                return `${d.date} (${d.day_name}) — ${d.days_out}d out`;
                            },
                            afterBody: ctx => {
                                const i = ctx[0].dataIndex;
                                const d = daily[i];
                                let lines = [
                                    `Demand: ${d.demand}`,
                                    `Weather: ${d.weather}`,
                                    `LF: ${(d.load_factor * 100).toFixed(0)}%`,
                                    `Competitor: \u20B9${d.competitor_price}`,
                                ];
                                if (d.event_name) lines.push(`Event: ${d.event_name}`);
                                return lines;
                            }
                        }
                    }
                }
            }
        });
    }

    function render365Monthly(D) {
        /* Group daily data by month */
        const months = {};
        D.daily.forEach(d => {
            const m = d.date.substring(0, 7); /* YYYY-MM */
            if (!months[m]) months[m] = [];
            months[m].push(d.avg_price);
        });
        const labels = Object.keys(months).map(m => {
            const dt = new Date(m + "-01");
            return dt.toLocaleDateString("en-IN", { month: "short", year: "2-digit" });
        });
        const avgByMonth = Object.values(months).map(arr => Math.round(arr.reduce((a, b) => a + b, 0) / arr.length));
        const overallAvg = Math.round(avgByMonth.reduce((a, b) => a + b, 0) / avgByMonth.length);
        const colors = avgByMonth.map(v => v >= overallAvg ? alpha(COLORS.red, 0.7) : alpha(COLORS.green, 0.7));
        const minVal = Math.min(...avgByMonth);

        make("c-365-monthly", {
            type: "bar",
            data: {
                labels: labels,
                datasets: [{
                    label: "Avg Price",
                    data: avgByMonth,
                    backgroundColor: colors,
                    borderRadius: 6,
                    maxBarThickness: 40,
                }]
            },
            options: {
                responsive: true, maintainAspectRatio: false,
                scales: {
                    x: { grid: { display: false } },
                    y: { ticks: { callback: v => "\u20B9" + INR(v) }, beginAtZero: false, suggestedMin: minVal * 0.85 }
                },
                plugins: {
                    legend: { display: false }
                }
            }
        });
    }

    function render365DoW(D) {
        const dowNames = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"];
        const dowBuckets = Array.from({ length: 7 }, () => []);
        D.daily.forEach(d => {
            const dt = new Date(d.date);
            dowBuckets[dt.getDay() === 0 ? 6 : dt.getDay() - 1].push(d.avg_price);
        });
        const dowAvg = dowBuckets.map(arr => arr.length ? Math.round(arr.reduce((a, b) => a + b, 0) / arr.length) : 0);
        const overallAvg = Math.round(dowAvg.reduce((a, b) => a + b, 0) / dowAvg.length);
        const barColors = dowAvg.map(v => v >= overallAvg ? alpha(COLORS.amber, 0.8) : alpha(COLORS.cyan, 0.7));
        const minDow = Math.min(...dowAvg.filter(v => v > 0));

        make("c-365-dow", {
            type: "bar",
            data: {
                labels: dowNames,
                datasets: [{
                    label: "Avg Price",
                    data: dowAvg,
                    backgroundColor: barColors,
                    borderRadius: 6,
                    maxBarThickness: 50,
                }]
            },
            options: {
                responsive: true, maintainAspectRatio: false,
                scales: {
                    x: { grid: { display: false } },
                    y: { ticks: { callback: v => "\u20B9" + INR(v) }, beginAtZero: false, suggestedMin: minDow * 0.85 }
                },
                plugins: { legend: { display: false } }
            }
        });
    }

    function render365EventTable(D) {
        /* Show notable days: events, holidays, extremes */
        const notable = D.daily.filter(d => d.event !== "none");
        /* Also add top-10 most expensive days */
        const sorted = [...D.daily].sort((a, b) => b.avg_price - a.avg_price);
        const top10 = sorted.slice(0, 10);
        const bottom10 = sorted.slice(-10).reverse();

        const renderRow = (d, tag) => {
            const badgeColor = d.event === "major" ? "danger" : d.event === "medium" ? "warning" : "secondary";
            const weatherIcon = d.weather === "storm" ? "\u26C8" : d.weather === "rain" ? "\uD83C\uDF27" : "\u2600";
            return `<tr>
                <td class="small fw-bold">${d.date}</td>
                <td class="small">${d.day_name}</td>
                <td><span class="badge bg-${badgeColor}">${d.event}</span></td>
                <td class="small">${d.event_name || "—"}</td>
                <td class="small">${weatherIcon} ${d.weather}</td>
                <td class="small">${(d.load_factor * 100).toFixed(0)}%</td>
                <td class="small">\u20B9${d.competitor_price}</td>
                <td class="fw-bold">\u20B9${INR(d.prices.leisure)}</td>
                <td class="fw-bold">\u20B9${INR(d.prices.business)}</td>
                <td class="fw-bold" style="color:#7c3aed">\u20B9${INR(d.avg_price)}</td>
                ${tag ? `<td><span class="badge bg-info">${tag}</span></td>` : "<td></td>"}
            </tr>`;
        };

        const wrap = document.getElementById("tbl-365-events");
        if (!wrap) return;
        wrap.innerHTML = `
            <ul class="nav nav-pills mb-2" id="tbl365tabs">
                <li class="nav-item"><button class="nav-link active small py-1 px-2" data-bs-toggle="pill" data-bs-target="#tbl365-events">Events & Festivals (${notable.length})</button></li>
                <li class="nav-item"><button class="nav-link small py-1 px-2" data-bs-toggle="pill" data-bs-target="#tbl365-peak">Top 10 Peak</button></li>
                <li class="nav-item"><button class="nav-link small py-1 px-2" data-bs-toggle="pill" data-bs-target="#tbl365-low">Top 10 Cheapest</button></li>
            </ul>
            <div class="tab-content">
                <div class="tab-pane show active" id="tbl365-events">
                    <table class="table table-sm table-hover mb-0" style="font-size:.82rem">
                        <thead><tr><th>Date</th><th>Day</th><th>Event</th><th>Name</th><th>Weather</th><th>Season</th><th>LF</th><th>Comp</th><th>Leisure</th><th>Business</th><th>Avg</th><th>Flags</th><th>Tag</th></tr></thead>
                        <tbody>${notable.map(d => renderRow(d, "")).join("")}</tbody>
                    </table>
                </div>
                <div class="tab-pane" id="tbl365-peak">
                    <table class="table table-sm table-hover mb-0" style="font-size:.82rem">
                        <thead><tr><th>Date</th><th>Day</th><th>Event</th><th>Name</th><th>Weather</th><th>Season</th><th>LF</th><th>Comp</th><th>Leisure</th><th>Business</th><th>Avg</th><th>Flags</th><th>Tag</th></tr></thead>
                        <tbody>${top10.map((d, i) => renderRow(d, "#" + (i + 1) + " Peak")).join("")}</tbody>
                    </table>
                </div>
                <div class="tab-pane" id="tbl365-low">
                    <table class="table table-sm table-hover mb-0" style="font-size:.82rem">
                        <thead><tr><th>Date</th><th>Day</th><th>Event</th><th>Name</th><th>Weather</th><th>Season</th><th>LF</th><th>Comp</th><th>Leisure</th><th>Business</th><th>Avg</th><th>Flags</th><th>Tag</th></tr></thead>
                        <tbody>${bottom10.map((d, i) => renderRow(d, "#" + (i + 1) + " Low")).join("")}</tbody>
                    </table>
                </div>
            </div>`;
    }

    /* =============================================================
       ML PREDICTIONS – User Input Forms
       ============================================================= */
    const INR_ML = v => "\u20B9" + Number(v).toLocaleString("en-IN");
    const NUM = v => Number(v).toLocaleString("en-IN");

    function riskBadge(r) {
        if (r === "High Risk") return '<span class="badge bg-danger fs-6">High Risk</span>';
        if (r === "Medium Risk") return '<span class="badge bg-warning text-dark fs-6">Medium Risk</span>';
        return '<span class="badge bg-success fs-6">Low Risk</span>';
    }

    /* Helper: collect inputs from a container div */
    function getInputs(containerId) {
        const wrap = document.getElementById(containerId);
        if (!wrap) return {};
        const obj = {};
        wrap.querySelectorAll("input, select").forEach(el => {
            const key = el.name || el.id;
            if (!key) return;
            const v = el.value;
            obj[key] = isNaN(v) || v === "" ? v : Number(v);
        });
        return obj;
    }

    /* Helper: post JSON and return result */
    async function mlPost(url, body, btn) {
        const origHTML = btn.innerHTML;
        btn.disabled = true;
        btn.innerHTML = '<span class="spinner-border spinner-border-sm me-1"></span>Predicting...';
        try {
            const res = await fetch(url, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(body),
            });
            if (!res.ok) { const e = await res.json(); throw new Error(e.detail || res.status); }
            return await res.json();
        } finally {
            btn.disabled = false;
            btn.innerHTML = origHTML;
        }
    }

    function resultCard(items, color) {
        return `<div class="alert alert-${color} py-2 px-3 mt-1 mb-2 d-flex flex-wrap gap-3 align-items-center" style="border-radius:10px;">
            ${items.map(([label, value]) => `<div><span class="text-muted small">${label}</span><br><span class="fw-bold fs-5">${value}</span></div>`).join('<div class="vr mx-1"></div>')}
        </div>`;
    }

    /* --- 1. DEMAND FORECASTING --- */
    const btnDemand = document.getElementById("btn-ml-demand");
    if (btnDemand) btnDemand.addEventListener("click", async () => {
        const d = getInputs("frm-demand");
        try {
            const r = await mlPost("/api/ml/predict/demand", d, btnDemand);
            document.getElementById("res-demand").innerHTML = resultCard([
                ["Route", r.route],
                ["Predicted Demand", `<span class="text-primary">${NUM(r.predicted_demand)}</span> passengers`],
            ], "primary");
        } catch (err) {
            document.getElementById("res-demand").innerHTML = `<div class="alert alert-danger py-2">${err.message}</div>`;
        }
    });

    /* --- 2. DYNAMIC PRICING --- */
    const btnPricing = document.getElementById("btn-ml-pricing");
    if (btnPricing) btnPricing.addEventListener("click", async () => {
        const d = getInputs("frm-pricing");
        try {
            const r = await mlPost("/api/ml/predict/pricing", d, btnPricing);
            document.getElementById("res-pricing").innerHTML = resultCard([
                ["Optimal Ticket Price", `<span class="text-success">${INR_ML(r.predicted_price)}</span>`],
            ], "success");
        } catch (err) {
            document.getElementById("res-pricing").innerHTML = `<div class="alert alert-danger py-2">${err.message}</div>`;
        }
    });

    /* --- 3. OVERBOOKING --- */
    const btnOverbook = document.getElementById("btn-ml-overbook");
    if (btnOverbook) btnOverbook.addEventListener("click", async () => {
        const d = getInputs("frm-overbook");
        try {
            const r = await mlPost("/api/ml/predict/overbooking", d, btnOverbook);
            let breakdownHTML = `<div class="table-responsive mt-2" style="max-height:200px">
                <table class="table table-sm table-striped mb-0"><thead><tr><th>Extra Seats</th><th>Denied Cost</th><th>Empty Cost</th><th>Total Cost</th></tr></thead><tbody>`;
            r.breakdown.forEach(b => {
                const hl = b.extra === r.optimal_extra_seats ? ' class="table-warning fw-bold"' : '';
                breakdownHTML += `<tr${hl}><td>${b.extra}</td><td>${INR_ML(b.denied_cost)}</td><td>${INR_ML(b.empty_cost)}</td><td>${INR_ML(b.total_cost)}</td></tr>`;
            });
            breakdownHTML += `</tbody></table></div>`;
            document.getElementById("res-overbook").innerHTML = resultCard([
                ["Optimal Extra Seats", `<span class="text-warning">${r.optimal_extra_seats}</span>`],
                ["Minimum Cost", INR_ML(r.min_cost)],
            ], "warning") + breakdownHTML;
        } catch (err) {
            document.getElementById("res-overbook").innerHTML = `<div class="alert alert-danger py-2">${err.message}</div>`;
        }
    });

    /* --- 4. ROUTE PROFITABILITY --- */
    const btnProfit = document.getElementById("btn-ml-profit");
    if (btnProfit) btnProfit.addEventListener("click", async () => {
        const d = getInputs("frm-profit");
        try {
            const r = await mlPost("/api/ml/predict/profitability", d, btnProfit);
            const mCls = r.margin > 90 ? "text-success" : r.margin > 50 ? "text-info" : "text-danger";
            document.getElementById("res-profit").innerHTML = resultCard([
                ["Predicted Profit", `<span class="text-info">${INR_ML(r.predicted_profit)}</span>`],
                ["Est. Revenue", INR_ML(r.estimated_revenue)],
                ["Margin", `<span class="${mCls}">${r.margin}%</span>`],
                ["R\u00B2 Score", r.r2_score],
            ], "info");
        } catch (err) {
            document.getElementById("res-profit").innerHTML = `<div class="alert alert-danger py-2">${err.message}</div>`;
        }
    });

    /* --- 5. OPERATIONAL RISK --- */
    const btnRisk = document.getElementById("btn-ml-risk");
    if (btnRisk) btnRisk.addEventListener("click", async () => {
        const d = getInputs("frm-risk");
        try {
            const r = await mlPost("/api/ml/predict/risk", d, btnRisk);
            let confHTML = Object.entries(r.confidence).map(([cat, pct]) => {
                const cls = cat === "High Risk" ? "danger" : cat === "Medium Risk" ? "warning" : "success";
                return `<div class="d-flex align-items-center gap-2 mb-1">
                    <span class="badge bg-${cls}" style="min-width:95px">${cat}</span>
                    <div class="progress flex-grow-1" style="height:8px"><div class="progress-bar bg-${cls}" style="width:${pct}%"></div></div>
                    <span class="fw-bold small" style="min-width:40px">${pct}%</span>
                </div>`;
            }).join("");
            document.getElementById("res-risk").innerHTML = resultCard([
                ["Predicted Risk", riskBadge(r.predicted_risk)],
                ["Model Accuracy", (r.accuracy * 100).toFixed(1) + "%"],
            ], "danger") + `<div class="px-1 mb-2"><div class="small fw-bold mb-1">Confidence Breakdown</div>${confHTML}</div>`;
        } catch (err) {
            document.getElementById("res-risk").innerHTML = `<div class="alert alert-danger py-2">${err.message}</div>`;
        }
    });

    /* --- 6. CUSTOMER CHURN PREDICTION --- */
    const btnChurn = document.getElementById("btn-ml-churn");
    if (btnChurn) btnChurn.addEventListener("click", async () => {
        const d = getInputs("frm-churn");
        try {
            const r = await mlPost("/api/ml/predict/churn", d, btnChurn);
            const riskCls = r.risk_level === "High Risk" ? "danger" : r.risk_level === "Medium Risk" ? "warning" : "success";
            const gaugeWidth = r.churn_probability;
            document.getElementById("res-churn").innerHTML = resultCard([
                ["Will Churn?", r.will_churn ? '<span class="badge bg-danger">YES</span>' : '<span class="badge bg-success">NO</span>'],
                ["Churn Probability", `<span class="fw-bold">${r.churn_probability}%</span>`],
                ["Risk Level", `<span class="badge bg-${riskCls}">${r.risk_level}</span>`],
                ["Accuracy", (r.model_metrics.accuracy * 100).toFixed(1) + "%"],
                ["F1 Score", (r.model_metrics.f1_score * 100).toFixed(1) + "%"],
                ["AUC-ROC", (r.model_metrics.auc_roc * 100).toFixed(1) + "%"],
            ], riskCls) + `
                <div class="px-1 mb-2">
                    <div class="small fw-bold mb-1">Churn Risk Gauge</div>
                    <div class="progress" style="height:14px">
                        <div class="progress-bar bg-${riskCls}" style="width:${gaugeWidth}%">${r.churn_probability}%</div>
                    </div>
                </div>
                <div class="alert alert-${riskCls} py-2 mb-2">
                    <i class="bi bi-lightbulb me-1"></i><strong>Recommendation:</strong> ${r.recommendation}
                </div>`;
        } catch (err) {
            document.getElementById("res-churn").innerHTML = `<div class="alert alert-danger py-2">${err.message}</div>`;
        }
    });

    /* --- 7. FLIGHT DELAY PREDICTION --- */
    const btnDelay = document.getElementById("btn-ml-delay");
    if (btnDelay) btnDelay.addEventListener("click", async () => {
        const d = getInputs("frm-delay");
        // Convert string booleans to actual booleans
        d.crew_delay = d.crew_delay === "true";
        d.technical_issue = d.technical_issue === "true";
        d.maintenance_flag = d.maintenance_flag === "true";
        try {
            const r = await mlPost("/api/ml/predict/delay", d, btnDelay);
            const delayCls = r.will_be_delayed ? "danger" : "success";
            let riskHTML = r.risk_factors.length > 0
                ? r.risk_factors.map(f => `<span class="badge bg-warning text-dark me-1 mb-1">${f}</span>`).join("")
                : '<span class="badge bg-success">No risk factors detected</span>';
            let impHTML = "";
            if (r.feature_importance && Object.keys(r.feature_importance).length > 0) {
                const sorted = Object.entries(r.feature_importance).sort((a, b) => b[1] - a[1]).slice(0, 5);
                impHTML = `<div class="small fw-bold mb-1 mt-2">Top Feature Importance</div>` +
                    sorted.map(([feat, imp]) => `<div class="d-flex align-items-center gap-2 mb-1">
                        <span class="small" style="min-width:180px">${feat}</span>
                        <div class="progress flex-grow-1" style="height:6px"><div class="progress-bar bg-warning" style="width:${(imp*100).toFixed(0)}%"></div></div>
                        <span class="small fw-bold">${(imp*100).toFixed(1)}%</span>
                    </div>`).join("");
            }
            document.getElementById("res-delay").innerHTML = resultCard([
                ["Will Be Delayed?", r.will_be_delayed ? '<span class="badge bg-danger">YES (>15 min)</span>' : '<span class="badge bg-success">ON TIME</span>'],
                ["Delay Probability", `<span class="fw-bold">${r.delay_probability}%</span>`],
                ["Model Accuracy", (r.model_accuracy * 100).toFixed(1) + "%"],
            ], delayCls) + `<div class="px-1 mb-2"><div class="small fw-bold mb-1">Risk Factors</div>${riskHTML}</div>` + `<div class="px-1 mb-2">${impHTML}</div>`;
        } catch (err) {
            document.getElementById("res-delay").innerHTML = `<div class="alert alert-danger py-2">${err.message}</div>`;
        }
    });

    /* --- 8. CANCELLATION PROBABILITY --- */
    const btnCancel = document.getElementById("btn-ml-cancel");
    if (btnCancel) btnCancel.addEventListener("click", async () => {
        const d = getInputs("frm-cancel");
        try {
            const r = await mlPost("/api/ml/predict/cancellation", d, btnCancel);
            const cancelCls = r.will_cancel ? "danger" : "success";
            let suggestHTML = r.mitigation_suggestions.length > 0
                ? r.mitigation_suggestions.map(s => `<li class="small">${s}</li>`).join("")
                : '<li class="small text-success">No mitigation needed</li>';
            document.getElementById("res-cancel").innerHTML = resultCard([
                ["Will Cancel?", r.will_cancel ? '<span class="badge bg-danger">LIKELY</span>' : '<span class="badge bg-success">UNLIKELY</span>'],
                ["Cancel Probability", `<span class="fw-bold">${r.cancellation_probability}%</span>`],
                ["Revenue at Risk", `<span class="fw-bold text-danger">₹${r.revenue_at_risk.toLocaleString()}</span>`],
                ["Accuracy", (r.model_metrics.accuracy * 100).toFixed(1) + "%"],
                ["F1 Score", (r.model_metrics.f1_score * 100).toFixed(1) + "%"],
                ["AUC-ROC", (r.model_metrics.auc_roc * 100).toFixed(1) + "%"],
            ], cancelCls) + `
                <div class="px-1 mb-2">
                    <div class="small fw-bold mb-1">Cancellation Risk</div>
                    <div class="progress" style="height:14px">
                        <div class="progress-bar bg-${cancelCls}" style="width:${r.cancellation_probability}%">${r.cancellation_probability}%</div>
                    </div>
                </div>
                <div class="alert alert-info py-2 mb-2">
                    <i class="bi bi-shield-check me-1"></i><strong>Mitigation Suggestions:</strong>
                    <ul class="mb-0 mt-1">${suggestHTML}</ul>
                </div>`;
        } catch (err) {
            document.getElementById("res-cancel").innerHTML = `<div class="alert alert-danger py-2">${err.message}</div>`;
        }
    });

    /* --- 9. LOAD FACTOR PREDICTION --- */
    const btnLF = document.getElementById("btn-ml-loadfactor");
    if (btnLF) btnLF.addEventListener("click", async () => {
        const d = getInputs("frm-loadfactor");
        try {
            const r = await mlPost("/api/ml/predict/load_factor", d, btnLF);
            const lfCls = r.risk_level === "High" ? "danger" : r.risk_level === "Medium" ? "warning" : "success";
            document.getElementById("res-loadfactor").innerHTML = resultCard([
                ["Predicted Load Factor", `<span class="fw-bold text-${lfCls}" style="font-size:1.4rem">${r.predicted_load_factor}%</span>`],
                ["Risk Level", `<span class="badge bg-${lfCls}">${r.risk_level}</span>`],
                ["R² Score", r.model_r2.toFixed(4)],
                ["MAPE", r.model_mape.toFixed(2) + "%"],
            ], lfCls) + `<div class="px-1 mb-2"><div class="progress" style="height:18px"><div class="progress-bar bg-${lfCls}" style="width:${r.predicted_load_factor}%">${r.predicted_load_factor}%</div></div></div>`;
        } catch (err) {
            document.getElementById("res-loadfactor").innerHTML = `<div class="alert alert-danger py-2">${err.message}</div>`;
        }
    });

    /* --- 10. NO-SHOW PREDICTION --- */
    const btnNS = document.getElementById("btn-ml-noshow");
    if (btnNS) btnNS.addEventListener("click", async () => {
        const d = getInputs("frm-noshow");
        try {
            const r = await mlPost("/api/ml/predict/noshow", d, btnNS);
            const nsCls = r.will_noshow ? "danger" : "success";
            document.getElementById("res-noshow").innerHTML = resultCard([
                ["Will No-Show?", r.will_noshow ? '<span class="badge bg-danger">YES</span>' : '<span class="badge bg-success">NO</span>'],
                ["No-Show Probability", `<span class="fw-bold">${r.noshow_probability}%</span>`],
                ["Accuracy", (r.model_metrics.accuracy * 100).toFixed(1) + "%"],
                ["F1 Score", (r.model_metrics.f1 * 100).toFixed(1) + "%"],
                ["AUC-ROC", (r.model_metrics.auc * 100).toFixed(1) + "%"],
            ], nsCls) + `<div class="px-1 mb-2"><div class="progress" style="height:14px"><div class="progress-bar bg-${nsCls}" style="width:${r.noshow_probability}%">${r.noshow_probability}%</div></div></div>`;
        } catch (err) {
            document.getElementById("res-noshow").innerHTML = `<div class="alert alert-danger py-2">${err.message}</div>`;
        }
    });

    /* --- 11. PASSENGER CLUSTERING --- */
    const btnCluster = document.getElementById("btn-ml-cluster");
    if (btnCluster) btnCluster.addEventListener("click", async () => {
        const d = getInputs("frm-cluster");
        try {
            const r = await mlPost("/api/ml/predict/cluster", d, btnCluster);
            let clusterList = Object.entries(r.all_clusters).map(([id, c]) => {
                const active = parseInt(id) === r.cluster_id ? "fw-bold text-primary" : "";
                return `<div class="d-flex justify-content-between ${active} mb-1"><span>${parseInt(id) === r.cluster_id ? "► " : ""}${c.name}</span><span class="badge bg-secondary">${c.size} pax</span></div>`;
            }).join("");
            document.getElementById("res-cluster").innerHTML = resultCard([
                ["Cluster", `<span class="badge fs-5" style="background:#6366f1">${r.cluster_name}</span>`],
                ["Cluster ID", r.cluster_id],
            ], "primary") + `<div class="row px-1 mb-2">
                <div class="col-md-6"><div class="small fw-bold mb-1">All Clusters</div>${clusterList}</div>
                <div class="col-md-6"><div class="small fw-bold mb-1">Cluster Profile</div>
                    ${Object.entries(r.cluster_profile).map(([k,v]) => `<div class="d-flex justify-content-between mb-1 small"><span>${k.replace(/_/g, ' ')}</span><span class="fw-bold">${v}</span></div>`).join("")}
                </div>
            </div>`;
        } catch (err) {
            document.getElementById("res-cluster").innerHTML = `<div class="alert alert-danger py-2">${err.message}</div>`;
        }
    });

    /* ---- Model 12: Dynamic Pricing Engine ---- */
    const btnDynPrice = document.getElementById("btn-dynprice");
    if (btnDynPrice) btnDynPrice.addEventListener("click", async () => {
        const d = getInputs("frm-dynprice");
        try {
            const r = await mlPost("/api/dynamic-price", d, btnDynPrice);
            const b = r.breakdown;
            const chgColor = r.price_change_pct >= 0 ? "#16a34a" : "#ef4444";
            const chgArrow = r.price_change_pct >= 0 ? "▲" : "▼";
            const barData = [
                { label: "Demand", val: b.demand.multiplier },
                { label: "Horizon", val: b.booking_horizon.multiplier },
                { label: "Event", val: b.event.multiplier },
                { label: "Weather", val: b.weather.multiplier },
                { label: "Competitor", val: b.competition.factor },
                { label: "Seat Load", val: b.seat_pressure.multiplier },
                { label: "Segment", val: b.segment.multiplier },
            ];
            const maxMul = Math.max(...barData.map(x => x.val), 1.35);
            const bars = barData.map(x => {
                const pct = (x.val / maxMul * 100).toFixed(0);
                const col = x.val > 1.01 ? "#16a34a" : x.val < 0.99 ? "#ef4444" : "#94a3b8";
                return `<div class="d-flex align-items-center mb-1">
                    <span class="small fw-bold" style="width:80px">${x.label}</span>
                    <div class="flex-grow-1 mx-2" style="height:18px;background:#f1f5f9;border-radius:4px;overflow:hidden">
                        <div style="width:${pct}%;height:100%;background:${col};border-radius:4px;transition:width .3s"></div>
                    </div>
                    <span class="small fw-bold" style="width:40px;text-align:right;color:${col}">${x.val.toFixed(2)}x</span>
                </div>`;
            }).join("");
            document.getElementById("res-dynprice").innerHTML = `
                <div class="row mb-3">
                    <div class="col-md-4 text-center">
                        <div class="small text-muted">Base Price</div>
                        <div class="fs-4 fw-bold">₹${INR_ML(r.base_price)}</div>
                    </div>
                    <div class="col-md-4 text-center">
                        <div class="small text-muted">Final Price</div>
                        <div class="fs-2 fw-bold" style="color:${chgColor}">₹${INR_ML(r.final_price)}</div>
                        ${r.clamped ? '<span class="badge bg-warning text-dark">Price clamped to limits</span>' : ''}
                    </div>
                    <div class="col-md-4 text-center">
                        <div class="small text-muted">Change</div>
                        <div class="fs-4 fw-bold" style="color:${chgColor}">${chgArrow} ${Math.abs(r.price_change_pct)}%</div>
                        <div class="small text-muted">Combined: ${r.combined_multiplier}x</div>
                    </div>
                </div>
                <div class="fw-bold small mb-2">Factor Breakdown</div>
                ${bars}
                <div class="text-muted mt-2" style="font-size:.7rem">Floor: ₹${INR_ML(r.constraints.min)} &middot; Ceiling: ₹${INR_ML(r.constraints.max)}</div>
            `;
        } catch (err) {
            document.getElementById("res-dynprice").innerHTML = `<div class="alert alert-danger py-2">${err.message}</div>`;
        }
    });

    /* ---- Boot ---- */
    document.addEventListener("DOMContentLoaded", init);
})();
