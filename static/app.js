const els = {
  trainFileInput: document.getElementById("trainFileInput"),
  predictFileInput: document.getElementById("predictFileInput"),
  trainFileName: document.getElementById("trainFileName"),
  predictFileName: document.getElementById("predictFileName"),
  downloadBtn: document.getElementById("downloadBtn"),
  statusText: document.getElementById("statusText"),
  serverState: document.getElementById("serverState"),
  featureList: document.getElementById("featureList"),
  currentDataRows: document.getElementById("currentDataRows"),
  currentDataColumns: document.getElementById("currentDataColumns"),
  datasetStepTitle: document.getElementById("datasetStepTitle"),
  stageKicker: document.getElementById("stageKicker"),
  stageTitle: document.getElementById("stageTitle"),
  stageSubtitle: document.getElementById("stageSubtitle"),
  datasetPreviewTitle: document.getElementById("datasetPreviewTitle"),
  datasetPreviewButton: document.getElementById("datasetPreviewButton"),
  datasetPreview: document.getElementById("datasetPreview"),
  predictPreview: document.getElementById("predictPreview"),
  predictNormalizedPreview: document.getElementById("predictNormalizedPreview"),
  basicStatsTable: document.getElementById("basicStatsTable"),
  statsTableTitle: document.getElementById("statsTableTitle"),
  boxplotTitle: document.getElementById("boxplotTitle"),
  boxplotSourceLabel: document.getElementById("boxplotSourceLabel"),
  normalityTable: document.getElementById("normalityTable"),
  outlierSummary: document.getElementById("outlierSummary"),
  outlierList: document.getElementById("outlierList"),
  trainingResults: document.getElementById("trainingResults"),
  predictionTable: document.getElementById("predictionTable"),
  modelResults: document.getElementById("modelResults"),
  boxplotImage: document.getElementById("boxplotImage"),
  distributionImage: document.getElementById("distributionImage"),
  correlationImage: document.getElementById("correlationImage"),
  activeEnvironmentLabel: document.getElementById("activeEnvironmentLabel"),
  activeDataRole: document.getElementById("activeDataRole"),
  describeSourceLabel: document.getElementById("describeSourceLabel"),
  environmentSwitch: document.getElementById("environmentSwitch"),
  modelConfigPanel: document.getElementById("modelConfigPanel"),
  variableSelect: document.getElementById("variableSelect"),
  variableSelectDistribution: document.getElementById("variableSelectDistribution"),
  modelChoice: document.getElementById("modelChoice"),
  modelChoicePredict: document.getElementById("modelChoicePredict"),
  runXgb: document.getElementById("runXgb"),
  runStacking: document.getElementById("runStacking"),
  xgbParams: document.getElementById("xgbParams"),
  stackingParams: document.getElementById("stackingParams"),
};

const steps = {
  train: ["Step 01", "训练数据确认", "选择内置示例数据，或上传带 yield 列的新农业数据来重新训练模型。", "panelTrain"],
  statsTable: ["Step 02", "统计表", "只对当前选择的数据生成描述统计表；训练数据和待预测数据不会混用。", "panelStatsTable"],
  boxplot: ["Step 03", "箱线图", "按所选变量查看当前数据的箱线图；训练数据和待预测数据分开计算。", "panelBoxplot"],
  distribution: ["Step 04", "变量诊断", "查看当前数据中所选变量的直方图、核密度曲线、Q-Q 图和正态性判断。", "panelDistribution"],
  outliers: ["Step 05", "清除异常值", "只对训练数据执行异常值清除，为后续模型训练准备更稳定的数据。", "panelOutliers"],
  correlation: ["Step 06", "相关性分析", "基于清除异常值后的训练数据生成相关性热力图。", "panelCorrelation"],
  trainModels: ["Step 07", "训练模型", "使用整份训练数据训练 XGBoost 和 Stacking 模型；这里不再切分网站测试集。", "panelTrainModels"],
  predictData: ["Step 05", "预测数据校验", "最终预测前检查待预测数据字段是否与模型特征一致。", "panelPredictData"],
  predict: ["Step 06", "产量预测", "使用训练数据已经训练好的模型，对待预测数据输出产量结果。", "panelPredict"],
};

let previewData = null;
let analysisData = null;
let predictPreviewData = null;
let latestPredictionRows = [];
let activeEnvironment = "train";
let requiredFeatureColumns = [];

function trainFile() {
  return els.trainFileInput.files[0] || null;
}

function predictFile() {
  return els.predictFileInput.files[0] || null;
}

function setStatus(text, isError = false) {
  els.statusText.textContent = text;
  els.statusText.classList.toggle("danger", isError);
}

function renderEmpty(target, text) {
  target.className = "table-wrap empty-state";
  target.textContent = text;
}

function envLabel() {
  return activeEnvironment === "train" ? "训练数据" : "待预测数据";
}

function setEnvironment(env, quiet = false) {
  activeEnvironment = env;
  els.environmentSwitch.querySelectorAll("button").forEach((btn) => {
    btn.classList.toggle("active", btn.dataset.env === env);
  });
  updateEnvironmentCopy();
  updateWorkflowVisibility();
  renderEnvironmentSummary();
  previewData = null;
  analysisData = null;
  resetCurrentMetrics();
  clearDatasetPreview();
  clearAnalysisViews();
  if (!quiet) setStatus(`已切换到${envLabel()}。`);
}

function datasetStepConfig() {
  if (activeEnvironment === "test") {
    return [
      "Step 01",
      "待预测数据确认",
      "上传待预测 CSV，并确认行数、字段、数值变量和预览内容。",
      "panelTrain",
    ];
  }
  return steps.train;
}

function analysisStepConfig(stepName) {
  const subject = activeEnvironment === "train" ? "训练数据" : "待预测数据";
  if (stepName === "statsTable") {
    return ["Step 02", `${subject}统计表`, `只统计${subject}本身，不引用另一类数据。`, "panelStatsTable"];
  }
  if (stepName === "boxplot") {
    return ["Step 03", `${subject}箱线图`, `按所选变量查看${subject}的箱线图，不与另一类数据混合。`, "panelBoxplot"];
  }
  if (stepName === "distribution") {
    return ["Step 04", `${subject}变量诊断`, `查看${subject}中所选变量的直方图、核密度曲线、Q-Q 图和正态性判断。`, "panelDistribution"];
  }
  return steps[stepName];
}

function getStepConfig(stepName) {
  if (stepName === "train") return datasetStepConfig();
  if (["statsTable", "boxplot", "distribution"].includes(stepName)) return analysisStepConfig(stepName);
  return steps[stepName];
}

function updateEnvironmentCopy() {
  const isTrain = activeEnvironment === "train";
  els.datasetStepTitle.textContent = isTrain ? "训练数据确认" : "待预测数据确认";
  els.datasetPreviewTitle.textContent = isTrain ? "训练数据预览" : "待预测数据预览";
  els.datasetPreviewButton.textContent = isTrain ? "载入训练数据" : "载入待预测数据";
  els.statsTableTitle.textContent = isTrain ? "训练数据统计表" : "待预测数据统计表";
  els.boxplotTitle.textContent = isTrain ? "训练数据箱线图" : "待预测数据箱线图";
  els.boxplotSourceLabel.textContent = isTrain ? "训练数据" : "待预测数据";
}

function updateWorkflowVisibility() {
  document.querySelectorAll("[data-env-scope]").forEach((el) => {
    el.classList.toggle("is-hidden", el.dataset.envScope !== activeEnvironment);
  });
  document.querySelectorAll(".workflow-nav .step-button:not(.is-hidden)").forEach((button, index) => {
    button.querySelector("span").textContent = String(index + 1).padStart(2, "0");
  });
  els.downloadBtn.classList.toggle("is-hidden", activeEnvironment !== "test");
}

function clearDatasetPreview() {
  renderEmpty(
    els.datasetPreview,
    activeEnvironment === "train" ? "等待载入训练数据" : "请上传待预测 CSV，并点击载入待预测数据"
  );
}

function resetCurrentMetrics() {
  els.currentDataRows.textContent = "-";
  els.currentDataColumns.textContent = "-";
}

function clearAnalysisViews() {
  renderTable(els.basicStatsTable, null);
  renderTable(els.normalityTable, null);
  [els.boxplotImage, els.distributionImage, els.correlationImage].forEach((img) => img.removeAttribute("src"));
}

function renderEnvironmentSummary() {
  if (activeEnvironment === "train") {
    els.activeEnvironmentLabel.textContent = "训练数据";
    els.activeDataRole.textContent = trainFile() ? `上传的新训练数据：${trainFile().name}` : "系统内置data";
    els.describeSourceLabel.textContent = "训练数据全部数值变量";
    return;
  }

  els.activeEnvironmentLabel.textContent = "待预测数据";
  els.activeDataRole.textContent = predictFile() ? `待预测数据：${predictFile().name}` : "等待上传待预测 CSV";
  els.describeSourceLabel.textContent = "待预测数据全部数值变量";
}

function setBusy(runName, busy) {
  document.querySelectorAll(".run-btn, .step-button").forEach((el) => (el.disabled = busy));
  const button = document.querySelector(`[data-run="${runName}"]`) || document.querySelector(`[data-step="${runName}"]`);
  if (button) button.classList.toggle("loading", busy);
}

function activateStep(stepName) {
  const requestedButton = document.querySelector(`[data-step="${stepName}"]`);
  if (requestedButton?.classList.contains("is-hidden")) {
    stepName = "train";
  }
  updateEnvironmentCopy();
  const [kicker, title, subtitle, panelId] = getStepConfig(stepName);
  const activeButton = document.querySelector(`[data-step="${stepName}"]`);
  els.stageKicker.textContent = activeButton ? `Step ${activeButton.querySelector("span").textContent}` : kicker;
  els.stageTitle.textContent = title;
  els.stageSubtitle.textContent = subtitle;

  document.querySelectorAll(".step-button").forEach((btn) => {
    btn.classList.toggle("active", btn.dataset.step === stepName);
  });
  document.querySelectorAll(".panel").forEach((panel) => {
    panel.classList.toggle("active", panel.id === panelId);
  });
}

function markDone(stepName) {
  document.querySelector(`[data-step="${stepName}"]`)?.classList.add("done");
}

function renderTable(target, tableData) {
  if (!tableData || !tableData.columns || !tableData.columns.length) {
    target.className = "table-wrap empty-state";
    target.textContent = "暂无内容";
    return;
  }

  const header = tableData.columns.map((col) => `<th>${col}</th>`).join("");
  const rows = tableData.rows
    .map((row) => `<tr>${tableData.columns.map((col) => `<td>${row[col] ?? ""}</td>`).join("")}</tr>`)
    .join("");

  target.className = "table-wrap";
  target.innerHTML = `<table><thead><tr>${header}</tr></thead><tbody>${rows}</tbody></table>`;
}

function renderFieldCheck(target, rows) {
  if (!rows.length) {
    renderEmpty(target, "暂无字段校验结果");
    return;
  }
  target.className = "table-wrap field-check-list";
  target.innerHTML = `
    <table>
      <thead>
        <tr>
          <th>必需字段</th>
          <th>是否在待预测数据中</th>
          <th>说明</th>
        </tr>
      </thead>
      <tbody>
        ${rows
          .map((item) => {
            const statusClass = item.present ? "status-ok" : "status-missing";
            const statusText = item.present ? "已包含" : "缺失";
            const note = item.present ? "可用于预测" : "需要补充该字段";
            return `<tr><td>${item.feature}</td><td class="${statusClass}">${statusText}</td><td>${note}</td></tr>`;
          })
          .join("")}
      </tbody>
    </table>
  `;
}

function updateVariableOptions(columns) {
  [els.variableSelect, els.variableSelectDistribution].forEach((select) => {
    select.innerHTML = "";
    columns.forEach((column) => {
      const option = document.createElement("option");
      option.value = column;
      option.textContent = column;
      select.appendChild(option);
    });
  });
}

function buildForm(includePredict = false) {
  const formData = new FormData();
  if (trainFile()) formData.append("train_file", trainFile());
  if ((includePredict || activeEnvironment === "test") && predictFile()) formData.append("predict_file", predictFile());
  formData.append("environment", activeEnvironment);
  const variable = document.getElementById("panelDistribution").classList.contains("active")
    ? els.variableSelectDistribution.value
    : els.variableSelect.value;
  const modelChoice = activeEnvironment === "test" ? els.modelChoicePredict.value : els.modelChoice.value;
  formData.append("variable", variable);
  formData.append("model_choice", modelChoice);
  formData.append("run_xgb", String(els.runXgb.checked));
  formData.append("run_stacking", String(els.runStacking.checked));
  formData.append("xgb_params", els.xgbParams.value);
  formData.append("stacking_params", els.stackingParams.value);
  return formData;
}

async function postForm(url, includePredict = false) {
  const response = await fetch(url, { method: "POST", body: buildForm(includePredict) });
  const payload = await response.json();
  if (!payload.ok) throw new Error(payload.error || "请求失败");
  return payload.data;
}

function renderReference(data) {
  requiredFeatureColumns = data.feature_columns || [];
  els.featureList.innerHTML = data.feature_columns.map((col) => `<span>${col}</span>`).join("");
  els.serverState.textContent = "服务已连接";
}

function renderTrainPreview(data) {
  previewData = data;
  els.currentDataRows.textContent = data.dataset.rows;
  els.currentDataColumns.textContent = data.dataset.columns;
  renderTable(els.datasetPreview, data.preview);
  updateVariableOptions(data.dataset.numeric_columns || []);
}

function renderStatsTable(descriptionData) {
  renderTable(els.basicStatsTable, descriptionData.basic_stats || descriptionData.stats);
  if (descriptionData?.dataset) {
    els.describeSourceLabel.textContent =
      activeEnvironment === "test"
        ? `待预测数据全部数值变量：${descriptionData.dataset.numeric_columns.length} 个`
        : `训练数据全部数值变量：${descriptionData.dataset.numeric_columns.length} 个`;
  }
}

function renderBoxplot(variableData) {
  if (variableData?.images?.boxplot) els.boxplotImage.src = variableData.images.boxplot;
}

function renderDistribution(data) {
  renderTable(els.normalityTable, data.normality);
  els.distributionImage.src = data.images.distribution;
}

function renderOutliers(data) {
  const preprocessing = data.preprocessing;
  if (!preprocessing) {
    els.outlierSummary.innerHTML = `<div class="summary-card"><span>状态</span><strong>训练数据缺少 yield</strong></div>`;
    renderEmpty(els.outlierList, "训练数据缺少 yield，无法清除异常值");
    return;
  }

  const cards = [
    ["原始训练数据", data.dataset.rows],
    ["参与建模数据", preprocessing.train_rows],
    ["清除后训练数据", preprocessing.cleaned_train_rows],
    ["删除异常行", preprocessing.removed_rows],
  ];
  els.outlierSummary.innerHTML = cards
    .map(([label, value]) => `<div class="summary-card"><span>${label}</span><strong>${value}</strong></div>`)
    .join("");
  const outlierRows = preprocessing.train_feature_outliers || [];
  if (!outlierRows.length) {
    renderEmpty(els.outlierList, "未发现需要清除的字段异常值");
    return;
  }
  renderTable(els.outlierList, {
    columns: ["字段", "清除行数", "清除方式", "判断依据", "下界", "上界"],
    rows: outlierRows.map((row) => ({
      字段: row.feature,
      清除行数: row.removed_rows ?? row.outliers,
      清除方式: row.method,
      判断依据: row.basis,
      下界: row.lower,
      上界: row.upper,
    })),
  });
}

function renderCorrelation(data) {
  if (data.images.correlation) els.correlationImage.src = data.images.correlation;
}

function renderPredictPreview(data) {
  predictPreviewData = data;
  els.currentDataRows.textContent = data.dataset.rows;
  els.currentDataColumns.textContent = data.dataset.columns;
  const requiredFeatures = data.dataset.required_features || requiredFeatureColumns;
  const missingFeatures = data.dataset.missing_features || [];
  const fieldCheck =
    data.field_check && data.field_check.length
      ? data.field_check
      : requiredFeatures.map((feature) => ({
          feature,
          present: !missingFeatures.includes(feature),
        }));
  renderFieldCheck(els.predictNormalizedPreview, fieldCheck);
  if (data.dataset.missing_features.length) {
    setStatus(`待预测数据缺少字段：${data.dataset.missing_features.join(", ")}`, true);
  } else {
    setStatus(`待预测数据字段校验通过：需要 ${requiredFeatures.length} 个字段，已全部包含。`);
  }
}

function renderTrainingResults(data) {
  const sourceLabels = {
    builtin_cached: "内置缓存模型",
    builtin_cache_created: "已生成内置缓存",
    trained_from_uploaded_data: "上传的新训练数据建模",
    trained_from_reference_data: "参考训练数据建模",
  };
  const summary = data.summary;
  const summaryCard = `
    <article class="model-card-result">
      <h3>训练数据</h3>
      <div class="metric-row"><span>训练行数</span><strong>${summary.training_rows}</strong></div>
      <div class="metric-row"><span>特征字段</span><strong>${summary.feature_count}</strong></div>
      <div class="metric-row"><span>网站逻辑</span><strong>全量训练</strong></div>
    </article>
  `;
  const modelCards = data.results
    .map(
      (item) => `
        <article class="model-card-result">
          <h3>${item.model}</h3>
          <div class="metric-row"><span>耗时</span><strong>${item.time_seconds}s</strong></div>
          <div class="metric-row"><span>输入</span><strong>整份训练数据</strong></div>
          <div class="metric-row"><span>模型来源</span><strong>${sourceLabels[item.source] || "当前训练"}</strong></div>
        </article>
      `
    )
    .join("");
  els.trainingResults.innerHTML = summaryCard + modelCards;
}

function renderPrediction(data) {
  latestPredictionRows = data.prediction_table.rows;
  els.currentDataRows.textContent = data.summary.new_rows;
  els.currentDataColumns.textContent = data.summary.feature_count;
  renderTable(els.predictionTable, data.prediction_table);
  const sourceLabels = {
    builtin_cached: "内置缓存模型",
    builtin_cache_created: "已生成内置缓存",
    trained_from_uploaded_data: "上传的新训练数据建模",
    trained_from_reference_data: "参考训练数据建模",
  };
  els.modelResults.innerHTML = data.results
    .map(
      (item) => `
        <article class="model-card-result">
          <h3>${item.model}</h3>
          <div class="metric-row"><span>耗时</span><strong>${item.time_seconds}s</strong></div>
          <div class="metric-row"><span>预测行数</span><strong>${item.predictions.length}</strong></div>
          <div class="metric-row"><span>模型来源</span><strong>${sourceLabels[item.source] || "当前训练"}</strong></div>
          <div class="metric-row"><span>输入</span><strong>农业生产指标</strong></div>
        </article>
      `
    )
    .join("");
  els.downloadBtn.disabled = latestPredictionRows.length === 0;
}

async function ensureTrainPreview() {
  if (previewData && previewData.environment === activeEnvironment) return previewData;
  const data = await postForm("/api/preview");
  data.environment = activeEnvironment;
  renderTrainPreview(data);
  markDone("train");
  return data;
}

async function runVariableAnalysis() {
  await ensureTrainPreview();
  return postForm("/api/variable", activeEnvironment === "test");
}

async function runDatasetDescription() {
  await ensureTrainPreview();
  return postForm("/api/describe", activeEnvironment === "test");
}

async function runAction(name) {
  setBusy(name, true);
  try {
    if (name === "train") {
      activateStep("train");
      setStatus(`正在载入${envLabel()}数据...`);
      const data = await postForm("/api/preview");
      data.environment = activeEnvironment;
      renderTrainPreview(data);
      markDone("train");
      setStatus(`${envLabel()}数据已载入，共 ${data.dataset.rows} 行。`);
    }

    if (name === "statsTable") {
      activateStep("statsTable");
      setStatus(`正在生成${envLabel()}统计表...`);
      const descriptionData = await runDatasetDescription();
      renderStatsTable(descriptionData);
      markDone("statsTable");
      setStatus(`${envLabel()}统计表已完成。`);
    }

    if (name === "boxplot") {
      activateStep("boxplot");
      setStatus(`正在生成${envLabel()}箱线图...`);
      const variableData = await runVariableAnalysis();
      renderBoxplot(variableData);
      markDone("boxplot");
      setStatus(`${envLabel()}箱线图已完成，当前变量：${variableData.variable}。`);
    }

    if (name === "distribution") {
      activateStep("distribution");
      setStatus(`正在生成${envLabel()}的变量分布诊断...`);
      const data = await runVariableAnalysis();
      renderDistribution(data);
      markDone("distribution");
      setStatus(`${data.variable} 的分布诊断已完成。`);
    }

    if (name === "outliers") {
      activateStep("outliers");
      if (activeEnvironment === "test") {
        throw new Error("异常值清除只对训练数据执行。待预测数据请先做预览、字段校验和产量预测。");
      }
      setStatus("正在清除训练数据异常值...");
      if (!analysisData) analysisData = await postForm("/api/analyze");
      const data = analysisData;
      renderOutliers(data);
      markDone("outliers");
      setStatus("异常值清除已完成。");
    }

    if (name === "correlation") {
      activateStep("correlation");
      if (activeEnvironment === "test") {
        throw new Error("相关性分析只对训练数据执行。待预测数据请先做预览、字段校验和产量预测。");
      }
      setStatus("正在生成训练数据相关性热力图...");
      if (!analysisData) analysisData = await postForm("/api/analyze");
      const data = analysisData;
      renderCorrelation(data);
      markDone("correlation");
      setStatus("相关性分析已完成。");
    }

    if (name === "trainModels") {
      activateStep("trainModels");
      if (activeEnvironment === "test") {
        throw new Error("模型训练只使用训练数据。待预测数据只用于最终产量预测。");
      }
      setStatus("正在用整份训练数据训练 XGBoost 和 Stacking 模型...");
      await ensureTrainPreview();
      const data = await postForm("/api/train-models");
      renderTrainingResults(data);
      markDone("trainModels");
      setStatus(`模型训练完成：训练行数 ${data.summary.training_rows}，特征字段 ${data.summary.feature_count} 个。`);
    }

    if (name === "predictData") {
      setEnvironment("test", true);
      activateStep("predictData");
      if (!predictFile()) throw new Error("请先上传待预测 CSV。");
      setStatus("正在校验待预测数据字段...");
      const data = await postForm("/api/predict-preview", true);
      renderPredictPreview(data);
      markDone("predictData");
    }

    if (name === "predict") {
      setEnvironment("test", true);
      activateStep("predict");
      if (!predictFile()) throw new Error("请先上传待预测 CSV。");
      const builtinTrain = !trainFile();
      setStatus(builtinTrain ? "正在使用训练数据已训练好的内置模型生成产量预测结果..." : "正在使用上传训练数据训练好的模型生成产量预测结果...");
      await ensureTrainPreview();
      const data = await postForm("/api/predict", true);
      renderPrediction(data);
      markDone("predict");
      setStatus(`产量预测完成，共 ${data.summary.new_rows} 行。`);
    }
  } catch (error) {
    setStatus(error.message, true);
  } finally {
    setBusy(name, false);
  }
}

function resetTraining() {
  previewData = null;
  analysisData = null;
  latestPredictionRows = [];
  els.downloadBtn.disabled = true;
  els.currentDataRows.textContent = "-";
  els.currentDataColumns.textContent = "-";
  clearDatasetPreview();
  renderTable(els.basicStatsTable, null);
  renderTable(els.normalityTable, null);
  [els.boxplotImage, els.distributionImage, els.correlationImage].forEach((img) => img.removeAttribute("src"));
  els.outlierSummary.innerHTML = "";
  renderEmpty(els.outlierList, "等待清除异常值");
  els.trainingResults.innerHTML = `<div class="placeholder-line"></div><div class="placeholder-line short"></div>`;
  document.querySelectorAll(".step-button").forEach((btn) => btn.classList.remove("done"));
}

function resetPrediction() {
  predictPreviewData = null;
  if (activeEnvironment === "test") previewData = null;
  latestPredictionRows = [];
  els.downloadBtn.disabled = true;
  els.currentDataRows.textContent = predictFile() ? "已选择" : "-";
  els.currentDataColumns.textContent = "-";
  if (activeEnvironment === "test") clearDatasetPreview();
  renderTable(els.predictPreview, null);
  renderTable(els.predictNormalizedPreview, null);
  renderTable(els.predictionTable, null);
  els.modelResults.innerHTML = `<div class="placeholder-line"></div><div class="placeholder-line short"></div>`;
  document.querySelector(`[data-step="predictData"]`)?.classList.remove("done");
  document.querySelector(`[data-step="predict"]`)?.classList.remove("done");
}

function toCsv(rows) {
  if (!rows.length) return "";
  const columns = Object.keys(rows[0]);
  return [
    columns.join(","),
    ...rows.map((row) =>
      columns
        .map((col) => {
          const value = row[col] ?? "";
          const text = String(value).replace(/"/g, '""');
          return /[",\n]/.test(text) ? `"${text}"` : text;
        })
        .join(",")
    ),
  ].join("\n");
}

function downloadCsv() {
  if (!latestPredictionRows.length) return;
  const blob = new Blob([toCsv(latestPredictionRows)], { type: "text/csv;charset=utf-8;" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = "predictions.csv";
  document.body.appendChild(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(url);
}

els.trainFileInput.addEventListener("change", () => {
  els.trainFileName.textContent = trainFile() ? trainFile().name : "系统内置data";
  renderEnvironmentSummary();
  resetTraining();
  setStatus("训练数据已变更，请重新载入；后续预测会使用这份训练数据训练好的模型。");
});

els.predictFileInput.addEventListener("change", () => {
  els.predictFileName.textContent = predictFile() ? predictFile().name : "请上传待预测 CSV";
  renderEnvironmentSummary();
  resetPrediction();
  setStatus("待预测数据已变更，请先校验字段。");
});

document.querySelectorAll(".step-button").forEach((button) => {
  button.addEventListener("click", () => activateStep(button.dataset.step));
});

document.querySelectorAll("[data-run]").forEach((button) => {
  button.addEventListener("click", () => runAction(button.dataset.run));
});

els.environmentSwitch.addEventListener("click", (event) => {
  const button = event.target.closest("[data-env]");
  if (!button) return;
  setEnvironment(button.dataset.env, true);
  activateStep("train");
  setStatus(`已切换到${envLabel()}。`);
});

els.variableSelect.addEventListener("change", () => {
  els.variableSelectDistribution.value = els.variableSelect.value;
});

els.variableSelectDistribution.addEventListener("change", () => {
  els.variableSelect.value = els.variableSelectDistribution.value;
});

els.modelChoice.addEventListener("change", () => {
  els.modelChoicePredict.value = els.modelChoice.value;
});

els.modelChoicePredict.addEventListener("change", () => {
  els.modelChoice.value = els.modelChoicePredict.value;
});

els.downloadBtn.addEventListener("click", downloadCsv);

fetch("/api/reference")
  .then((res) => res.json())
  .then((data) => {
    renderReference(data);
    updateEnvironmentCopy();
    updateWorkflowVisibility();
    renderEnvironmentSummary();
    return runAction("train");
  })
  .catch(() => {
    els.serverState.textContent = "连接失败";
    setStatus("无法连接后端服务。", true);
  });
