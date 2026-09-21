/**
 * Customer Churn Intelligence Portal — Application Client Logic
 * Interacts with Flask API: /api/metrics, /api/customers, /api/customers/:id/analyze, /api/customers/:id/simulate
 */

// Application State
const state = {
  searchQuery: "",
  riskFilter: "all",
  contractFilter: "all",
  internetFilter: "all",
  sortBy: "churn_prob",
  sortDir: "desc",
  page: 1,
  perPage: 25,
  totalCustomers: 0,
  totalPages: 1,
  selectedCustomerIndex: null,
  activeCustomerAnalysis: null
};

// DOM Element References
const elements = {
  // KPIs
  kpiTotal: document.getElementById("kpi-total"),
  kpiPredictedChurn: document.getElementById("kpi-predicted-churn"),
  kpiPredictedChurnPct: document.getElementById("kpi-predicted-churn-pct"),
  kpiHighRisk: document.getElementById("kpi-high-risk"),
  kpiLowRisk: document.getElementById("kpi-low-risk"),
  kpiAccuracy: document.getElementById("kpi-accuracy"),

  // Search & Filters
  searchInput: document.getElementById("search-input"),
  searchClearBtn: document.getElementById("search-clear-btn"),
  riskFilterGroup: document.getElementById("risk-filter-group"),
  contractFilter: document.getElementById("contract-filter"),
  internetFilter: document.getElementById("internet-filter"),
  sortSelect: document.getElementById("sort-select"),
  resultsCount: document.getElementById("results-count"),
  activeFilterBadge: document.getElementById("active-filter-badge"),

  // Table & Pagination
  customerTbody: document.getElementById("customer-tbody"),
  perPageSelect: document.getElementById("per-page-select"),
  prevPageBtn: document.getElementById("prev-page-btn"),
  nextPageBtn: document.getElementById("next-page-btn"),
  currentPageNum: document.getElementById("current-page-num"),
  totalPagesNum: document.getElementById("total-pages-num"),

  // Drawer
  drawer: document.getElementById("analysis-drawer"),
  drawerBackdrop: document.getElementById("drawer-backdrop"),
  drawerCloseBtn: document.getElementById("drawer-close-btn"),
  drawerCustId: document.getElementById("drawer-cust-id"),
  drawerRiskBadge: document.getElementById("drawer-risk-badge"),
  drawerVerdictBadge: document.getElementById("drawer-verdict-badge"),
  drawerActualBadge: document.getElementById("drawer-actual-badge"),
  drawerBody: document.getElementById("drawer-body")
};

// Initialize App
document.addEventListener("DOMContentLoaded", () => {
  setupEventListeners();
  loadMetrics();
  loadCustomers();
});

function setupEventListeners() {
  // Debounced search
  let searchTimeout = null;
  elements.searchInput.addEventListener("input", (e) => {
    const val = e.target.value;
    elements.searchClearBtn.style.display = val ? "block" : "none";
    clearTimeout(searchTimeout);
    searchTimeout = setTimeout(() => {
      state.searchQuery = val;
      state.page = 1;
      loadCustomers();
    }, 250);
  });

  elements.searchClearBtn.addEventListener("click", () => {
    elements.searchInput.value = "";
    elements.searchClearBtn.style.display = "none";
    state.searchQuery = "";
    state.page = 1;
    loadCustomers();
  });

  // Risk filter pills
  elements.riskFilterGroup.addEventListener("click", (e) => {
    const btn = e.target.closest(".pill-btn");
    if (!btn) return;
    elements.riskFilterGroup.querySelectorAll(".pill-btn").forEach(b => b.classList.remove("active"));
    btn.classList.add("active");
    state.riskFilter = btn.dataset.risk;
    state.page = 1;
    loadCustomers();
  });

  // Dropdown filters
  elements.contractFilter.addEventListener("change", (e) => {
    state.contractFilter = e.target.value;
    state.page = 1;
    loadCustomers();
  });

  elements.internetFilter.addEventListener("change", (e) => {
    state.internetFilter = e.target.value;
    state.page = 1;
    loadCustomers();
  });

  elements.sortSelect.addEventListener("change", (e) => {
    const [by, dir] = e.target.value.split("-");
    state.sortBy = by;
    state.sortDir = dir;
    state.page = 1;
    loadCustomers();
  });

  // Pagination controls
  elements.perPageSelect.addEventListener("change", (e) => {
    state.perPage = parseInt(e.target.value, 10);
    state.page = 1;
    loadCustomers();
  });

  elements.prevPageBtn.addEventListener("click", () => {
    if (state.page > 1) {
      state.page--;
      loadCustomers();
    }
  });

  elements.nextPageBtn.addEventListener("click", () => {
    if (state.page < state.totalPages) {
      state.page++;
      loadCustomers();
    }
  });

  // Drawer dismissals
  elements.drawerCloseBtn.addEventListener("click", closeDrawer);
  elements.drawerBackdrop.addEventListener("click", closeDrawer);
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && elements.drawer.classList.contains("open")) {
      closeDrawer();
    }
  });
}

// --------------------------------------------------------------------------
// API Calls
// --------------------------------------------------------------------------

async function loadMetrics() {
  try {
    const res = await fetch("/api/metrics");
    const json = await res.json();
    if (json.status === "success") {
      const data = json.data;
      elements.kpiTotal.textContent = data.total_customers.toLocaleString();
      elements.kpiPredictedChurn.textContent = data.predicted_churn_count.toLocaleString();
      
      const churnPct = ((data.predicted_churn_count / data.total_customers) * 100).toFixed(1);
      elements.kpiPredictedChurnPct.textContent = `${churnPct}% of pool`;

      elements.kpiHighRisk.textContent = data.high_risk_count.toLocaleString();
      elements.kpiLowRisk.textContent = data.low_risk_count.toLocaleString();
      elements.kpiAccuracy.textContent = `${data.accuracy_pct}%`;
    }
  } catch (err) {
    console.error("Error loading metrics:", err);
  }
}

async function loadCustomers() {
  // Show table loader
  elements.customerTbody.innerHTML = `
    <tr>
      <td colspan="11" class="loading-cell">
        <div class="spinner"></div>
        <span>Searching evaluation pool...</span>
      </td>
    </tr>
  `;

  try {
    const params = new URLSearchParams({
      q: state.searchQuery,
      risk: state.riskFilter,
      contract: state.contractFilter,
      internet: state.internetFilter,
      sort_by: state.sortBy,
      sort_dir: state.sortDir,
      page: state.page,
      per_page: state.perPage
    });

    const res = await fetch(`/api/customers?${params.toString()}`);
    const json = await res.json();

    if (json.status === "success") {
      const data = json.data;
      state.totalCustomers = data.total;
      state.totalPages = data.total_pages;

      renderTable(data.customers);
      updatePagination();
    } else {
      elements.customerTbody.innerHTML = `<tr><td colspan="11" class="loading-cell">Error: ${json.message}</td></tr>`;
    }
  } catch (err) {
    console.error("Error loading customers:", err);
    elements.customerTbody.innerHTML = `<tr><td colspan="11" class="loading-cell">Failed to connect to backend server.</td></tr>`;
  }
}

function renderTable(customers) {
  elements.resultsCount.textContent = state.totalCustomers.toLocaleString();

  if (state.searchQuery || state.riskFilter !== "all" || state.contractFilter !== "all" || state.internetFilter !== "all") {
    elements.activeFilterBadge.style.display = "inline-block";
    elements.activeFilterBadge.textContent = "Filtered";
  } else {
    elements.activeFilterBadge.style.display = "none";
  }

  if (!customers || customers.length === 0) {
    elements.customerTbody.innerHTML = `
      <tr>
        <td colspan="11" class="loading-cell">
          <span>No customers matching current filter criteria.</span>
        </td>
      </tr>
    `;
    return;
  }

  const rowsHtml = customers.map(cust => {
    const isHigh = cust.risk_tier === "High";
    const isMed = cust.risk_tier === "Medium";
    const tierClass = isHigh ? "high" : (isMed ? "med" : "low");

    const probPct = cust.churn_prob_pct.toFixed(1);
    const actualClass = cust.actual_churn === 1 ? "churned" : "retained";
    const actualText = cust.actual_churn === 1 ? "Actual: Churn" : "Actual: Stay";

    const isSelected = state.selectedCustomerIndex === cust.test_index;

    return `
      <tr class="${isSelected ? 'row-selected' : ''}" data-index="${cust.test_index}">
        <td class="cell-id mono">${cust.customerID}</td>
        <td>
          <div class="cell-risk-meter">
            <div class="risk-bar-track">
              <div class="risk-bar-fill ${tierClass}" style="width: ${Math.min(100, Math.max(5, cust.churn_prob * 100))}%"></div>
            </div>
            <span class="prob-number ${tierClass}">${probPct}%</span>
          </div>
        </td>
        <td>
          <span class="badge-chip ${tierClass}">${cust.prediction_label}</span>
        </td>
        <td>
          <span class="badge-actual ${actualClass}">${actualText}</span>
        </td>
        <td class="tenure-cell">${cust.tenure} mo</td>
        <td>${cust.Contract}</td>
        <td>${cust.InternetService}</td>
        <td>${cust.PaymentMethod}</td>
        <td class="text-right mono">$${cust.MonthlyCharges.toFixed(2)}</td>
        <td class="text-right mono">$${cust.TotalCharges.toFixed(2)}</td>
        <td class="text-center">
          <button class="action-inspect-btn" data-index="${cust.test_index}">Analyze &rarr;</button>
        </td>
      </tr>
    `;
  }).join("");

  elements.customerTbody.innerHTML = rowsHtml;

  // Bind click on rows and buttons
  elements.customerTbody.querySelectorAll("tr").forEach(row => {
    row.addEventListener("click", (e) => {
      const idx = parseInt(row.dataset.index, 10);
      openCustomerAnalysis(idx);
    });
  });
}

function updatePagination() {
  elements.currentPageNum.textContent = state.page;
  elements.totalPagesNum.textContent = state.totalPages || 1;
  elements.prevPageBtn.disabled = state.page <= 1;
  elements.nextPageBtn.disabled = state.page >= state.totalPages;
}

// --------------------------------------------------------------------------
// Customer Analysis Drawer Logic
// --------------------------------------------------------------------------

async function openCustomerAnalysis(testIndex) {
  state.selectedCustomerIndex = testIndex;

  // Update selected row in table
  elements.customerTbody.querySelectorAll("tr").forEach(r => {
    if (parseInt(r.dataset.index, 10) === testIndex) {
      r.classList.add("row-selected");
    } else {
      r.classList.remove("row-selected");
    }
  });

  // Open drawer
  elements.drawerBackdrop.classList.add("active");
  elements.drawer.classList.add("open");
  elements.drawer.setAttribute("aria-hidden", "false");

  elements.drawerCustId.textContent = `Loading #${testIndex}...`;
  elements.drawerBody.innerHTML = `
    <div class="drawer-loading">
      <div class="spinner"></div>
      <span>Querying model pipeline &amp; tracing decision tree...</span>
    </div>
  `;

  try {
    const res = await fetch(`/api/customers/${testIndex}/analyze`);
    const json = await res.json();
    if (json.status === "success") {
      state.activeCustomerAnalysis = json.data;
      renderAnalysis(json.data);
    } else {
      elements.drawerBody.innerHTML = `<div class="drawer-loading">Failed to analyze customer: ${json.message}</div>`;
    }
  } catch (err) {
    console.error("Error analyzing customer:", err);
    elements.drawerBody.innerHTML = `<div class="drawer-loading">Error connecting to server.</div>`;
  }
}

function closeDrawer() {
  elements.drawer.classList.remove("open");
  elements.drawer.setAttribute("aria-hidden", "true");
  elements.drawerBackdrop.classList.remove("active");
  state.selectedCustomerIndex = null;
  elements.customerTbody.querySelectorAll("tr.row-selected").forEach(r => r.classList.remove("row-selected"));
}

function renderAnalysis(data) {
  const cust = data.summary;
  const isHigh = cust.risk_tier === "High";
  const isMed = cust.risk_tier === "Medium";
  const tierClass = isHigh ? "high" : (isMed ? "med" : "low");

  // Update header badges
  elements.drawerCustId.textContent = cust.customerID;
  elements.drawerRiskBadge.className = `risk-badge badge-chip ${tierClass}`;
  elements.drawerRiskBadge.textContent = `${cust.risk_tier.toUpperCase()} RISK`;

  elements.drawerVerdictBadge.textContent = cust.prediction_label;
  elements.drawerActualBadge.textContent = cust.actual_churn === 1 ? "Actual Ground Truth: Churned" : "Actual Ground Truth: Retained";
  elements.drawerActualBadge.className = `actual-badge badge-actual ${cust.actual_churn === 1 ? 'churned' : 'retained'}`;

  // SVG Gauge calculations (radius = 42, circumference = 2 * PI * 42 ≈ 263.89)
  const radius = 42;
  const circumference = 2 * Math.PI * radius;
  const offset = circumference - (cust.churn_prob * circumference);
  const strokeColor = isHigh ? "var(--risk-high-border)" : (isMed ? "var(--risk-med-border)" : "var(--risk-low-border)");

  // Render main drawer body sections
  elements.drawerBody.innerHTML = `
    <!-- Section 1: Probability Gauge & Diagnostic Summary -->
    <div class="dossier-card">
      <div class="dossier-section-title">
        <span>Probability Assessment</span>
        <span class="mono">${cust.prediction_label}</span>
      </div>
      <div class="prob-meter-layout">
        <div class="svg-gauge-box">
          <svg class="svg-gauge" viewBox="0 0 100 100">
            <circle class="gauge-bg" cx="50" cy="50" r="${radius}"></circle>
            <circle class="gauge-progress" cx="50" cy="50" r="${radius}"
              stroke="${strokeColor}"
              stroke-dasharray="${circumference}"
              stroke-dashoffset="${offset}"></circle>
          </svg>
          <div class="gauge-center-text">
            <span class="gauge-percent">${cust.churn_prob_pct}%</span>
            <span class="gauge-sublabel">Churn Risk</span>
          </div>
        </div>
        <div class="verdict-details">
          <h3 class="verdict-headline" style="color: ${strokeColor}">
            ${isHigh ? 'High Risk of Imminent Attrition' : (isMed ? 'Moderate Watchlist Customer' : 'Strong Retention Baseline')}
          </h3>
          <p class="verdict-desc">
            ${isHigh 
              ? 'The DecisionTree model detected high-churn signals such as month-to-month contracting, electronic billing, or low account tenure.'
              : 'Customer exhibits stabilizing contract or tenure attributes. Ongoing service quality will sustain retention.'}
          </p>
        </div>
      </div>
    </div>

    <!-- Section 2: Account Specifications Grid -->
    <div class="dossier-card">
      <div class="dossier-section-title">
        <span>Account Dossier</span>
        <span class="mono">Profile Snapshot</span>
      </div>
      <div class="spec-grid">
        <div class="spec-item">
          <span class="spec-key">Contract Term</span>
          <span class="spec-val">${cust.Contract}</span>
        </div>
        <div class="spec-item">
          <span class="spec-key">Account Tenure</span>
          <span class="spec-val">${cust.tenure} months</span>
        </div>
        <div class="spec-item">
          <span class="spec-key">Monthly Spend</span>
          <span class="spec-val mono">$${cust.MonthlyCharges.toFixed(2)}</span>
        </div>
        <div class="spec-item">
          <span class="spec-key">Total Historical Spend</span>
          <span class="spec-val mono">$${cust.TotalCharges.toFixed(2)}</span>
        </div>
        <div class="spec-item">
          <span class="spec-key">Internet Connection</span>
          <span class="spec-val">${cust.InternetService}</span>
        </div>
        <div class="spec-item">
          <span class="spec-key">Payment Method</span>
          <span class="spec-val">${cust.PaymentMethod}</span>
        </div>
        <div class="spec-item">
          <span class="spec-key">Tech Support Bundle</span>
          <span class="spec-val">${cust.TechSupport}</span>
        </div>
        <div class="spec-item">
          <span class="spec-key">Online Security</span>
          <span class="spec-val">${cust.OnlineSecurity}</span>
        </div>
        <div class="spec-item">
          <span class="spec-key">Senior Citizen</span>
          <span class="spec-val">${cust.SeniorCitizen === 1 ? 'Yes' : 'No'}</span>
        </div>
        <div class="spec-item">
          <span class="spec-key">Partner / Dependents</span>
          <span class="spec-val">${cust.Partner} / ${cust.Dependents}</span>
        </div>
      </div>
    </div>

    <!-- Section 3: Model Signal Impact (Features evaluated by pipeline) -->
    <div class="dossier-card">
      <div class="dossier-section-title">
        <span>Model Signals &amp; Importance</span>
        <span class="mono">Pipeline CTF Selection</span>
      </div>
      <div class="signals-list">
        ${data.feature_signals.map(sig => `
          <div class="signal-row">
            <div class="signal-header">
              <span class="signal-name">${formatFeatureName(sig.feature)}</span>
              <span class="signal-impact-tag ${sig.risk_contribution.includes('Risk') ? 'elevated' : 'protective'}">
                ${sig.risk_contribution}
              </span>
            </div>
            <div class="signal-bar-track">
              <div class="signal-bar-fill" style="width: ${Math.max(8, sig.importance_pct * 1.5)}%"></div>
            </div>
            <div class="signal-meta">
              <span>Customer Value: <strong>${sig.customer_value}</strong></span>
              <span>Model Weight: ${sig.importance_pct}%</span>
            </div>
          </div>
        `).join("")}
      </div>
    </div>

    <!-- Section 4: Decision Tree Step-by-Step Explanation Trace -->
    <div class="dossier-card">
      <div class="dossier-section-title">
        <span>Decision Path Trace</span>
        <span class="mono">Exact Tree Walk (${data.decision_path.length} steps)</span>
      </div>
      <div class="tree-path-steps">
        ${data.decision_path.map((step, idx) => {
          const isLeaf = step.node_type.includes("Leaf");
          return `
            <div class="path-step-node ${isLeaf ? 'step-leaf' : ''}">
              <div class="step-meta">
                <span>Node #${step.node_id} &bull; ${step.node_type}</span>
                <span>${step.class_dist}</span>
              </div>
              <div class="step-rule">
                ${isLeaf ? `Final Destination &rarr; Prediction: ${step.prediction === 1 ? 'Churn' : 'Retain'}` : `Split: ${formatFeatureName(step.feature)} ${step.threshold_rule}`}
              </div>
              <div class="step-verdict">
                ${isLeaf ? `Leaf Samples: <strong>${step.samples}</strong> &bull; Node Churn Risk: <strong>${step.churn_prob_pct}%</strong>` : `Customer value is <strong>${step.customer_value}</strong> &rarr; Branch taken: <strong>${step.direction}</strong>`}
              </div>
            </div>
          `;
        }).join("")}
      </div>
    </div>

    <!-- Section 5: Interactive What-If Intervention Simulator -->
    <div class="dossier-card">
      <div class="dossier-section-title">
        <span>What-If Intervention Simulator</span>
        <span class="mono">Interactive Test</span>
      </div>
      <div class="simulator-box">
        <p style="font-size: 12px; color: var(--text-secondary);">
          Simulate contract or payment restructuring to observe how the model re-evaluates this customer's risk profile.
        </p>

        <div class="sim-field-group">
          <label for="sim-contract">Simulate Contract Change:</label>
          <select id="sim-contract">
            <option value="Month-to-month" ${cust.Contract === 'Month-to-month' ? 'selected' : ''}>Month-to-month</option>
            <option value="One year" ${cust.Contract === 'One year' ? 'selected' : ''}>One year</option>
            <option value="Two year" ${cust.Contract === 'Two year' ? 'selected' : ''}>Two year (Retention Lock)</option>
          </select>
        </div>

        <div class="sim-field-group">
          <label for="sim-payment">Simulate Payment Method:</label>
          <select id="sim-payment">
            <option value="Electronic check" ${cust.PaymentMethod === 'Electronic check' ? 'selected' : ''}>Electronic check</option>
            <option value="Mailed check" ${cust.PaymentMethod === 'Mailed check' ? 'selected' : ''}>Mailed check</option>
            <option value="Bank transfer (automatic)" ${cust.PaymentMethod === 'Bank transfer (automatic)' ? 'selected' : ''}>Bank transfer (automatic)</option>
            <option value="Credit card (automatic)" ${cust.PaymentMethod === 'Credit card (automatic)' ? 'selected' : ''}>Credit card (automatic)</option>
          </select>
        </div>

        <button class="sim-run-btn" id="sim-run-btn">Run Intervention Simulation &rarr;</button>

        <div id="sim-result-container" style="display:none;"></div>
      </div>
    </div>

    <!-- Section 6: Actionable Retention Playbook -->
    <div class="dossier-card">
      <div class="dossier-section-title">
        <span>Retention Recommendations</span>
        <span class="mono">Prescriptive Actions</span>
      </div>
      <div class="playbook-actions">
        ${data.retention_actions.map(act => `
          <div class="playbook-card ${act.priority.toLowerCase()}">
            <div class="playbook-card-title">${act.title} (${act.priority} Priority)</div>
            <div class="playbook-card-action">${act.action}</div>
            <div class="playbook-card-reason">${act.reason}</div>
          </div>
        `).join("")}
      </div>
    </div>
  `;

  // Attach simulator event listener
  const simBtn = document.getElementById("sim-run-btn");
  if (simBtn) {
    simBtn.addEventListener("click", runSimulation);
  }
}

async function runSimulation() {
  const simContract = document.getElementById("sim-contract").value;
  const simPayment = document.getElementById("sim-payment").value;
  const resultBox = document.getElementById("sim-result-container");

  resultBox.style.display = "block";
  resultBox.innerHTML = `
    <div class="sim-result-strip">
      <div class="spinner"></div>
      <span>Re-running pipeline for simulated attributes...</span>
    </div>
  `;

  try {
    const res = await fetch(`/api/customers/${state.selectedCustomerIndex}/simulate`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        contract: simContract,
        payment_method: simPayment
      })
    });

    const json = await res.json();
    if (json.status === "success") {
      const sim = json.data;
      const isBetter = sim.reduced_risk;
      const badgeClass = isBetter ? "reduced" : "increased";
      const sign = sim.delta_pct <= 0 ? "" : "+";

      resultBox.innerHTML = `
        <div class="sim-result-strip">
          <div>
            <strong>Simulated Churn Risk: ${sim.simulated_prob_pct}%</strong>
            <div style="font-size: 11px; color: var(--text-muted); margin-top: 2px;">
              Original: ${sim.original_prob_pct}% &bull; Verdict: ${sim.simulated_label}
            </div>
          </div>
          <span class="delta-badge ${badgeClass}">
            ${sign}${sim.delta_pct}%
          </span>
        </div>
      `;
    } else {
      resultBox.innerHTML = `<div class="sim-result-strip">Simulation error: ${json.message}</div>`;
    }
  } catch (err) {
    console.error("Simulation error:", err);
    resultBox.innerHTML = `<div class="sim-result-strip">Failed to execute simulation.</div>`;
  }
}

function formatFeatureName(feat) {
  if (feat === "tenure") return "Account Tenure (Months)";
  if (feat === "InternetService_Fiber optic") return "Fiber Optic Internet";
  if (feat === "Contract_Two year") return "Two-Year Contract";
  if (feat === "PaymentMethod_Electronic check") return "Electronic Check Billing";
  return feat.replace(/_/g, " ");
}
