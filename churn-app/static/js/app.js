/**
 * Telco Churn Intelligence & Retention Offer Engine
 * Frontend Client Controller
 */

document.addEventListener('DOMContentLoaded', () => {
  // Runtime State
  const state = {
    page: 1,
    limit: 15,
    search: '',
    status: 'churn', // Start by showing High Risk Churners as requested
    contract: 'all',
    internet: 'all',
    totalItems: 0,
    totalPages: 1,
    currentModalCustomer: null,
    currentModalOffer: null
  };

  // DOM Elements
  const tableBody = document.getElementById('tableBody');
  const searchInput = document.getElementById('searchInput');
  const btnClearSearch = document.getElementById('btnClearSearch');
  const filterStatus = document.getElementById('filterStatus');
  const filterContract = document.getElementById('filterContract');
  const filterInternet = document.getElementById('filterInternet');
  const pageSizeSelect = document.getElementById('pageSizeSelect');
  const btnResetFilters = document.getElementById('btnResetFilters');

  const btnPrevPage = document.getElementById('btnPrevPage');
  const btnNextPage = document.getElementById('btnNextPage');
  const currentPageNum = document.getElementById('currentPageNum');
  const totalPageNum = document.getElementById('totalPageNum');
  const pageStart = document.getElementById('pageStart');
  const pageEnd = document.getElementById('pageEnd');
  const totalRecords = document.getElementById('totalRecords');

  // KPI elements
  const kpiTotal = document.getElementById('kpiTotal');
  const kpiChurn = document.getElementById('kpiChurn');
  const kpiChurnRate = document.getElementById('kpiChurnRate');
  const kpiRetained = document.getElementById('kpiRetained');
  const kpiAnalyzed = document.getElementById('kpiAnalyzed');
  const kpiAnalyzedBar = document.getElementById('kpiAnalyzedBar');
  const kpiOffersSent = document.getElementById('kpiOffersSent');
  const btnBatchAnalyzeTop = document.getElementById('btnBatchAnalyzeTop');

  // Modal elements
  const modal = document.getElementById('analysisModal');
  const btnModalClose = document.getElementById('btnModalClose');
  const modalCustId = document.getElementById('modalCustId');
  const modalPredBadge = document.getElementById('modalPredBadge');
  const mpTenure = document.getElementById('mpTenure');
  const mpContract = document.getElementById('mpContract');
  const mpInternet = document.getElementById('mpInternet');
  const mpPayment = document.getElementById('mpPayment');
  const mpMonthly = document.getElementById('mpMonthly');
  const mpTotal = document.getElementById('mpTotal');
  const nodeStepperContainer = document.getElementById('nodeStepperContainer');

  const offerBadge = document.getElementById('offerBadge');
  const offerTitle = document.getElementById('offerTitle');
  const offerDiscount = document.getElementById('offerDiscount');
  const offerRiskFactor = document.getElementById('offerRiskFactor');
  const offerIncentivesList = document.getElementById('offerIncentivesList');
  const offerRationale = document.getElementById('offerRationale');
  const offerRevenueSaved = document.getElementById('offerRevenueSaved');
  const btnSendOffer = document.getElementById('btnSendOffer');
  const offerStatusIndicator = document.getElementById('offerStatusIndicator');
  const offerStatusText = document.getElementById('offerStatusText');

  const toastContainer = document.getElementById('toastContainer');

  // =========================================================================
  // API Fetch Functions
  // =========================================================================

  async function fetchCustomers() {
    renderLoadingState();
    try {
      const params = new URLSearchParams({
        page: state.page,
        limit: state.limit,
        search: state.search,
        status: state.status,
        contract: state.contract,
        internet: state.internet
      });

      const res = await fetch(`/api/customers?${params.toString()}`);
      const data = await res.json();

      if (data.success) {
        state.totalItems = data.pagination.total_items;
        state.totalPages = data.pagination.total_pages;
        state.page = data.pagination.page;
        renderTable(data.customers);
        renderPagination();
      } else {
        renderErrorState(data.error || 'Failed to load customers');
      }
    } catch (err) {
      console.error(err);
      renderErrorState('Network connection error');
    }
  }

  async function updateKPIs() {
    try {
      const res = await fetch('/api/stats');
      const data = await res.json();
      if (data.success && data.stats) {
        const s = data.stats;
        kpiTotal.textContent = s.total.toLocaleString();
        kpiChurn.textContent = s.churn_count.toLocaleString();
        kpiChurnRate.textContent = `${s.churn_rate}%`;
        kpiRetained.textContent = s.retained_count.toLocaleString();
        kpiAnalyzed.textContent = s.analyzed_count.toLocaleString();
        kpiOffersSent.textContent = s.offers_sent_count.toLocaleString();

        const pct = s.total > 0 ? Math.round((s.analyzed_count / s.total) * 100) : 0;
        kpiAnalyzedBar.style.width = `${pct}%`;
      }
    } catch (e) {
      console.warn('Failed to refresh stats:', e);
    }
  }

  // =========================================================================
  // Table Rendering
  // =========================================================================

  function renderTable(customers) {
    if (!customers || customers.length === 0) {
      tableBody.innerHTML = `
        <tr>
          <td colspan="9" class="text-center py-6" style="padding: 40px; color: var(--text-muted);">
            No customers match the specified filter criteria.
          </td>
        </tr>
      `;
      return;
    }

    tableBody.innerHTML = customers.map(c => {
      const isChurn = c.pred === 1;
      const probPct = Math.round(c.prob * 100);
      const riskClass = isChurn ? 'risk-high' : 'risk-low';
      const riskBadgeClass = isChurn ? 'badge-churn' : 'badge-retained';
      const riskLabel = isChurn ? `Churn (${probPct}%)` : `Retained (${100 - probPct}%)`;

      let workflowBadge = `<span class="badge badge-pending">Pending</span>`;
      if (c.offer_sent) {
        workflowBadge = `<span class="badge badge-sent">Offer Dispatched</span>`;
      } else if (c.analyzed) {
        workflowBadge = `<span class="badge badge-analyzed">Analyzed</span>`;
      }

      return `
        <tr data-customer-id="${c.customerID}">
          <td><span class="cust-id-badge">${c.customerID}</span></td>
          <td><strong>${c.tenure}</strong> mos</td>
          <td>${c.Contract}</td>
          <td>${c.InternetService}</td>
          <td><span style="font-size: 12px; color: var(--text-secondary);">${c.PaymentMethod}</span></td>
          <td class="font-mono"><strong>$${c.MonthlyCharges.toFixed(2)}</strong></td>
          <td>
            <div class="risk-pill">
              <div class="risk-meter" title="Probability: ${probPct}%">
                <div class="risk-meter-fill ${riskClass}" style="width: ${probPct}%"></div>
              </div>
              <span class="badge ${riskBadgeClass}">${riskLabel}</span>
            </div>
          </td>
          <td>${workflowBadge}</td>
          <td class="text-right">
            <button class="btn btn-secondary btn-sm btn-analyze" data-id="${c.customerID}">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" style="width: 14px; height: 14px;">
                <circle cx="11" cy="11" r="8"></circle>
                <line x1="21" y1="21" x2="16.65" y2="16.65"></line>
              </svg>
              <span>${c.analyzed ? 'View Trace' : 'Analyze'}</span>
            </button>
          </td>
        </tr>
      `;
    }).join('');

    // Bind Analyze Buttons
    document.querySelectorAll('.btn-analyze').forEach(btn => {
      btn.addEventListener('click', (e) => {
        const id = e.currentTarget.getAttribute('data-id');
        analyzeCustomer(id);
      });
    });
  }

  function renderLoadingState() {
    tableBody.innerHTML = `
      <tr>
        <td colspan="9" style="text-align: center; padding: 40px;">
          <div class="spinner"></div>
          <p style="margin-top: 10px; color: var(--text-secondary); font-size: 13px;">Loading customer records...</p>
        </td>
      </tr>
    `;
  }

  function renderErrorState(msg) {
    tableBody.innerHTML = `
      <tr>
        <td colspan="9" style="text-align: center; padding: 30px; color: var(--color-rose);">
          Error: ${msg}
        </td>
      </tr>
    `;
  }

  function renderPagination() {
    currentPageNum.textContent = state.page;
    totalPageNum.textContent = state.totalPages;
    totalRecords.textContent = state.totalItems.toLocaleString();

    const start = state.totalItems === 0 ? 0 : (state.page - 1) * state.limit + 1;
    const end = Math.min(state.page * state.limit, state.totalItems);
    pageStart.textContent = start;
    pageEnd.textContent = end;

    btnPrevPage.disabled = state.page <= 1;
    btnNextPage.disabled = state.page >= state.totalPages;
  }

  // =========================================================================
  // Customer Analysis & Decision Tree Traversal Modal
  // =========================================================================

  async function analyzeCustomer(customerId) {
    showToast('Analyzing Profile', `Tracing Decision Tree path for ${customerId}...`, 'info');

    try {
      const res = await fetch(`/api/analyze/${customerId}`, { method: 'POST' });
      const data = await res.json();

      if (!data.success) {
        showToast('Error', data.error || 'Failed to analyze customer', 'warning');
        return;
      }

      state.currentModalCustomer = data.customer;
      state.currentModalOffer = data.offer;

      // Populate Modal Header & Profile
      modalCustId.textContent = `ID: ${data.customer.customerID}`;
      const isChurn = data.customer.pred === 1;
      const probPct = Math.round(data.customer.prob * 100);

      modalPredBadge.className = `badge ${isChurn ? 'badge-churn' : 'badge-retained'}`;
      modalPredBadge.textContent = isChurn ? `HIGH CHURN RISK (${probPct}%)` : `LOW RISK - RETAINED (${100 - probPct}%)`;

      const raw = data.customer.raw;
      mpTenure.textContent = `${raw.tenure} months`;
      mpContract.textContent = raw.Contract;
      mpInternet.textContent = raw.InternetService;
      mpPayment.textContent = raw.PaymentMethod;
      mpMonthly.textContent = `$${parseFloat(raw.MonthlyCharges).toFixed(2)}/mo`;
      mpTotal.textContent = `$${raw.TotalCharges}`;

      // Populate Decision Tree Node Stepper
      renderNodeStepper(data.path, isChurn);

      // Populate Retention Offer Card
      const offer = data.offer;
      offerBadge.textContent = offer.badge;
      offerTitle.textContent = offer.title;
      offerDiscount.textContent = offer.discount;
      offerRiskFactor.textContent = offer.key_risk_factor;
      offerRationale.textContent = offer.rationale;
      offerRevenueSaved.textContent = offer.est_revenue_saved;

      offerIncentivesList.innerHTML = offer.incentives.map(inc => `<li>${inc}</li>`).join('');

      // Dispatch Status
      if (data.customer.offer_sent) {
        offerStatusIndicator.style.display = 'flex';
        offerStatusText.textContent = `Offer Dispatched (${data.customer.offer_sent_at})`;
        btnSendOffer.disabled = true;
        btnSendOffer.style.opacity = '0.5';
        btnSendOffer.innerHTML = `<span>Offer Already Dispatched</span>`;
      } else {
        offerStatusIndicator.style.display = 'none';
        btnSendOffer.disabled = false;
        btnSendOffer.style.opacity = '1';
        btnSendOffer.innerHTML = `
          <svg class="btn-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <line x1="22" y1="2" x2="11" y2="13"></line>
            <polygon points="22 2 15 22 11 13 2 9 22 2"></polygon>
          </svg>
          <span>Dispatch Retention Offer</span>
        `;
      }

      // Show Modal
      modal.style.display = 'flex';

      // Update UI row and KPIs
      updateRowState(data.customer);
      updateKPIs();

    } catch (e) {
      console.error(e);
      showToast('Error', 'Analysis request failed', 'warning');
    }
  }

  function renderNodeStepper(nodes, isChurn) {
    if (!nodes || nodes.length === 0) {
      nodeStepperContainer.innerHTML = `
        <div style="padding: 16px; background: rgba(15, 23, 42, 0.6); border-radius: 8px; color: var(--text-secondary); font-size: 13px;">
          Direct branch classification (Leaf reached at root evaluation).
        </div>
      `;
      return;
    }

    nodeStepperContainer.innerHTML = nodes.map((n, idx) => {
      const isLeaf = n.node_type === 'Leaf' || idx === nodes.length - 1;
      const condClass = n.condition_met ? 'cond-true' : 'cond-false';
      const condText = n.condition_met ? '✓ Condition Met (Left Branch)' : '✕ Condition Not Met (Right Branch)';

      return `
        <div class="node-card">
          <span class="node-dot ${isLeaf ? 'dot-leaf' : ''}"></span>
          <div class="node-card-header">
            <span class="node-title">${idx === 0 ? 'Root Node 0' : (isLeaf ? `Leaf Node ${n.node_id}` : `Decision Branch Node ${n.node_id}`)}</span>
            <span class="badge ${n.condition_met ? 'badge-analyzed' : 'badge-pending'}">${n.node_type || 'Split'}</span>
          </div>
          <div class="node-rule">${n.threshold_rule || n.feature}</div>
          <div class="node-meta-row">
            <span>Customer Value: <strong style="color: #fff;">${n.customer_value}</strong></span>
            <span class="condition-badge ${condClass}">${condText}</span>
          </div>
        </div>
      `;
    }).join('');
  }

  function updateRowState(customer) {
    const row = document.querySelector(`tr[data-customer-id="${customer.customerID}"]`);
    if (!row) return;

    // Update workflow status column (index 7)
    const statusCell = row.children[7];
    if (customer.offer_sent) {
      statusCell.innerHTML = `<span class="badge badge-sent">Offer Dispatched</span>`;
    } else {
      statusCell.innerHTML = `<span class="badge badge-analyzed">Analyzed</span>`;
    }

    // Update button text
    const btn = row.querySelector('.btn-analyze span');
    if (btn) btn.textContent = 'View Trace';
  }

  // =========================================================================
  // Retention Offer Dispatch Action
  // =========================================================================

  btnSendOffer.addEventListener('click', async () => {
    if (!state.currentModalCustomer) return;
    const cid = state.currentModalCustomer.customerID;

    btnSendOffer.disabled = true;
    btnSendOffer.innerHTML = `<div class="spinner" style="width: 16px; height: 16px; border-width: 2px;"></div> Dispatching...`;

    try {
      const res = await fetch(`/api/send-offer/${cid}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          offer_title: state.currentModalOffer ? state.currentModalOffer.title : 'Custom Package',
          channel: 'Automated Multichannel (Email + SMS)'
        })
      });

      const data = await res.json();
      if (data.success) {
        state.currentModalCustomer.offer_sent = true;
        state.currentModalCustomer.offer_sent_at = data.details.sent_at;

        offerStatusIndicator.style.display = 'flex';
        offerStatusText.textContent = `Offer Dispatched (${data.details.sent_at})`;

        btnSendOffer.innerHTML = `<span>Offer Successfully Dispatched!</span>`;
        btnSendOffer.style.opacity = '0.5';

        showToast('Retention Offer Dispatched', `Package '${data.details.offer_title}' sent to ${cid} via ${data.details.channel}`, 'success');

        updateRowState(state.currentModalCustomer);
        updateKPIs();
      } else {
        btnSendOffer.disabled = false;
        btnSendOffer.innerHTML = `<span>Retry Dispatch</span>`;
        showToast('Error', data.error || 'Dispatch failed', 'warning');
      }
    } catch (e) {
      console.error(e);
      btnSendOffer.disabled = false;
      btnSendOffer.innerHTML = `<span>Retry Dispatch</span>`;
      showToast('Error', 'Dispatch request failed', 'warning');
    }
  });

  // =========================================================================
  // Batch Analyze Action
  // =========================================================================

  btnBatchAnalyzeTop.addEventListener('click', async () => {
    btnBatchAnalyzeTop.disabled = true;
    btnBatchAnalyzeTop.innerHTML = `
      <div class="spinner" style="width: 16px; height: 16px; border-width: 2px;"></div>
      <span>Batch Analyzing (1,409)...</span>
    `;

    try {
      const res = await fetch('/api/batch-analyze', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({})
      });

      const data = await res.json();
      if (data.success) {
        showToast('Batch Analysis Complete', data.message, 'success');
        updateKPIs();
        fetchCustomers();
      }
    } catch (e) {
      console.error(e);
      showToast('Error', 'Batch analyze failed', 'warning');
    } finally {
      btnBatchAnalyzeTop.disabled = false;
      btnBatchAnalyzeTop.innerHTML = `
        <svg class="btn-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"></polygon>
        </svg>
        <span class="btn-text">Batch Analyze All (1,409)</span>
      `;
    }
  });

  // =========================================================================
  // Modal Open/Close Controls
  // =========================================================================

  btnModalClose.addEventListener('click', () => {
    modal.style.display = 'none';
  });

  window.addEventListener('click', (e) => {
    if (e.target === modal) {
      modal.style.display = 'none';
    }
  });

  window.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && modal.style.display === 'flex') {
      modal.style.display = 'none';
    }
  });

  // =========================================================================
  // Filter & Search Event Handlers
  // =========================================================================

  let searchTimeout = null;
  searchInput.addEventListener('input', (e) => {
    const val = e.target.value.trim();
    btnClearSearch.style.display = val ? 'block' : 'none';

    clearTimeout(searchTimeout);
    searchTimeout = setTimeout(() => {
      state.search = val;
      state.page = 1;
      fetchCustomers();
    }, 280);
  });

  btnClearSearch.addEventListener('click', () => {
    searchInput.value = '';
    btnClearSearch.style.display = 'none';
    state.search = '';
    state.page = 1;
    fetchCustomers();
  });

  filterStatus.addEventListener('change', (e) => {
    state.status = e.target.value;
    state.page = 1;
    fetchCustomers();
  });

  filterContract.addEventListener('change', (e) => {
    state.contract = e.target.value;
    state.page = 1;
    fetchCustomers();
  });

  filterInternet.addEventListener('change', (e) => {
    state.internet = e.target.value;
    state.page = 1;
    fetchCustomers();
  });

  pageSizeSelect.addEventListener('change', (e) => {
    state.limit = parseInt(e.target.value, 10);
    state.page = 1;
    fetchCustomers();
  });

  btnResetFilters.addEventListener('click', () => {
    searchInput.value = '';
    btnClearSearch.style.display = 'none';
    filterStatus.value = 'churn';
    filterContract.value = 'all';
    filterInternet.value = 'all';
    pageSizeSelect.value = '15';

    state.search = '';
    state.status = 'churn';
    state.contract = 'all';
    state.internet = 'all';
    state.limit = 15;
    state.page = 1;

    fetchCustomers();
  });

  // Pagination clicks
  btnPrevPage.addEventListener('click', () => {
    if (state.page > 1) {
      state.page--;
      fetchCustomers();
    }
  });

  btnNextPage.addEventListener('click', () => {
    if (state.page < state.totalPages) {
      state.page++;
      fetchCustomers();
    }
  });

  // =========================================================================
  // Toast System
  // =========================================================================

  function showToast(title, message, type = 'info') {
    const toast = document.createElement('div');
    toast.className = `toast toast-${type}`;
    toast.innerHTML = `
      <div>
        <div class="toast-title">${title}</div>
        <div class="toast-message">${message}</div>
      </div>
    `;

    toastContainer.appendChild(toast);

    setTimeout(() => {
      toast.style.opacity = '0';
      toast.style.transform = 'translateY(10px)';
      toast.style.transition = 'all 0.3s ease';
      setTimeout(() => toast.remove(), 300);
    }, 4500);
  }

  // =========================================================================
  // Initialize
  // =========================================================================
  fetchCustomers();
  updateKPIs();
});
