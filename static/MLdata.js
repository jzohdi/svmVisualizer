// goToPage(SVMmethod, chartId, data.Id);
let myChartCreated = false;
let defaultAnimation = "easeInQuint";
let CURRENT_DATA = "Sample";
const sizeFor3dPoint = 10;
const sizeFor2dPoint = 15;

const goToPage = (SVMmethod, chartId, dataId) => {
  const url =
    "/svm_plot?params=" +
    JSON.stringify({ method: SVMmethod, chart: chartId, data: dataId });
  window.open(url, "_blank");
};

const sizeFor2DPoint = () => {
  const windowSize = window.innerWidth;
  return windowSize / 100;
};

const sizeFor3DPoint = () => {
  const windowSize = window.innerWidth;
  return (windowSize * 2) / 100;
};

const deepCopy = obj => {
  return JSON.parse(JSON.stringify(obj));
};

const printf = array_of_values => {
  array_of_values.forEach(value => {
    console.log(value);
  });
};
const mapData = (num, min_One, max_One, min_Out, max_Out) => {
  return (
    ((num - min_One) * (max_Out - min_Out)) / (max_One - min_One) + min_Out
  );
};

const string = data => {
  return JSON.stringify(data);
};
const representData = [
  { color: "rgba(255, 99, 132, alpha)", label: "Classification 1" },
  { color: "rgba(0, 124, 249, alpha)", label: "Classification 2" },
  { color: "rgba(25, 150, 100, alpha)", label: "Classification 3" },
  { color: "rgba(93, 1, 150, alpha)", label: "Classification 4" }
];

const randomRGBA = _ => {
  var o = Math.round,
    r = Math.random,
    s = 255;
  return "rgba(" + o(r() * s) + "," + o(r() * s) + "," + o(r() * s) + ",alpha)";
};

const getNextColor = nextIndex => {
  if (nextIndex < representData.length) {
    return representData[nextIndex].color;
  }
  return randomRGBA();
};

const getColorForValue = (usedColorsDict, value) => {
  if (usedColorsDict.hasOwnProperty(value)) {
    return usedColorsDict[value];
  }
  usedColorsDict[value] = getNextColor(Object.keys(usedColorsDict).length);
  return usedColorsDict[value];
};

const mapConfidence = (confidenceSet, index) => {
  if (confidenceSet == null) {
    return false;
  }
  return mapData(confidenceSet[index], 0.49, 1, 0.5, 0.99);
};

const createNew2dTrace = value => {
  return {
    x: [],
    y: [],
    mode: "markers",
    name: value,
    text: [],
    marker: {
      size: sizeFor2dPoint,
      color: []
    },
    type: "scatter"
  };
};

const createNew3dTrace = value => {
  return {
    x: [],
    y: [],
    z: [],
    mode: "markers",
    name: value,
    text: [],
    marker: {
      size: sizeFor3dPoint,
      color: [],
      // opacity: [],
      symbol: "circle"
    },
    type: "scatter3d"
  };
};

const getTraceForValue = (dictionaryOfTraces, value, dimensions) => {
  if (dictionaryOfTraces.hasOwnProperty(value)) {
    return dictionaryOfTraces[value];
  }
  if (dimensions === 3) {
    dictionaryOfTraces[value] = createNew3dTrace(value);
    return dictionaryOfTraces[value];
  }
  if (dimensions === 2) {
    dictionaryOfTraces[value] = createNew2dTrace(value);
    return dictionaryOfTraces[value];
  }
  console.log("getTraceForValue used incorrectly, args passed in: ", {
    traces: dictionaryOfTraces,
    value: value,
    dimensions: dimensions
  });
};

const parse2dData = data_set => {
  const colorForValue = {};
  const traces = {};

  data_set.result.forEach((value, index) => {
    const trace = getTraceForValue(traces, value, 2);
    const color = getColorForValue(colorForValue, value);
    // const confidence = mapConfidence(data_set["confidence"], index);
    const formattedRGBA = color.replace("alpha", "1");
    trace.x.push(data_set.test_data[index][0]);
    trace.y.push(data_set.test_data[index][1]);
    trace.text.push("value: " + value);
    trace.marker.color.push(formattedRGBA);
  });

  return Object.values(traces);
};

/**
 * takes in 3d data json and returns array of data points used in creating
 * chart js. the data returned must be parsed to create the correct input
 * for Plotly object
 *
 * @param {object} data_set  data_set json returned by retrieveModel
 * @returns {array} finalData array of data points;
 */
const parse3dData = data_set => {
  const colorForValue = {};
  const traces = {};

  data_set.result.forEach((value, index) => {
    const trace = getTraceForValue(traces, value, 3);
    const color = getColorForValue(colorForValue, value);
    // const confidence = mapConfidence(data_set["confidence"], index);
    const formattedRGBA = color.replace(
      "alpha",
      "1" // confidence.toFixed(2).toString()
    );
    trace.x.push(data_set.test_data[index][0]);
    trace.y.push(data_set.test_data[index][1]);
    trace.z.push(data_set.test_data[index][2]);
    trace.text.push("value: " + value);
    trace.marker.color.push(formattedRGBA);
  });

  return Object.values(traces);
};

const cacheData = {};

const runMethod = chartId => {
  console.log("running method...");
  const selectedText = $("#select-method :selected").text(); // The text content of the selected option
  const selectedVal = $("#select-method").val();
  if (cacheData.hasOwnProperty(selectedVal) && CURRENT_DATA === "Sample") {
    createScatter(cacheData[selectedVal].plot, chartId);
  } else {
    showModel(selectedVal, chartId);
  }
};

const parseModelData = (data, SVMmethod, chartId) => {
  delete data["status"];
  const bestParams = data["params"];
  delete data["params"];
  const bestScore = data["score"];
  delete data["score"];

  if (data.test_data[0].length === 2) {
    dataSet = parse2dData(data);
  }

  if (data.test_data[0].length === 3) {
    dataSet = parse3dData(data);
  }

  if (CURRENT_DATA === "Sample") {
    cacheData[SVMmethod] = {
      plot: dataSet,
      score: bestScore,
      params: bestParams
    };
  }

  createScatter(dataSet, chartId);
  console.log("method call successful");
};

const showSamplePlot = (SVMmethod, chartId, model_id) => {
  // Do retrive model once at beginning to check if the data set is cached in data base.
  $.get("/retrieve_model/" + model_id).done(function(data) {
    if (data["status"] === "Finished") {
      parseModelData(data, SVMmethod, chartId);
    } else if (data.hasOwnProperty("error")) {
      alert(data.error);
    } else {
      alert("Something went wrong getting sample data.");
    }
  });
};

const initSampleData = () => {
  const SVMmethod = "Sample Data";
  const chartId = "myChart";
  $.post("/train_model/", {
    runMethod: SVMmethod,
    data_set: CURRENT_DATA === "Sample" ? CURRENT_DATA : string(CURRENT_DATA)
  })
    .done(function(data) {
      if (data.Status == "Success") {
        showSamplePlot(SVMmethod, chartId, data.Id);
      } else {
        alert("Something went wrong.");
      }
    })
    .fail(function(error) {
      alert(error);
    });
};

const showModel = (SVMmethod, chartId) => {
  // console.log(CURRENT_DATA);
  $.post("/train_model/", {
    runMethod: SVMmethod,
    data_set: CURRENT_DATA === "Sample" ? CURRENT_DATA : string(CURRENT_DATA)
  })
    .done(function(data) {
      if (data.Status == "Success") {
        goToPage(SVMmethod, chartId, data.Id);
      } else {
        alert("Something went wrong.");
      }
    })
    .fail(function(error) {
      alert(error);
    });
};
const getChartWidth = () => {
  if (window.innerWidth < 1200) {
    return window.innerWidth * 0.85;
  } else {
    return window.innerWidth * 0.7;
  }
};

const calculateDimension = () => {
  return { d: getChartWidth() };
};
const charts = {};
const layout = {
  title: "",
  autosize: true,
  hovermode: "closest",
  showlegend: false,
  width: window.innerWidth - 100,
  height: window.innerWidth - 100,
  legend: {
    x: 0.3,
    y: 1.1
  },
  xaxis: {
    zeroline: false,
    hoverformat: ".3f"
  },
  yaxis: {
    zeroline: false,
    hoverformat: ".3r"
  }
};

const createScatter = (dataSet, targetCanvas) => {
  const dimensions = calculateDimension();
  layout.width = dimensions.d;
  layout.height = dimensions.d;
  // layout["margin-left"] = dimensions["margin-left"];
  Plotly.newPlot(targetCanvas, dataSet, layout, { resonsive: true });
};

const runData = () => {
  const selectedData = $("#select-data :selected").text();

  if (selectedData === "Sample Data") {
    CURRENT_DATA = "Sample";
    const sampleData = cacheData["Sample Data"].plot;
    createScatter(sampleData, "myChart");
  } else if (selectedData === "Manual Data") {
    const manualData = parseManualData();

    if (manualData[0] == undefined) {
      $("#manual-data-error").html(
        "Could not parse data, check that all rows are filled appropriately."
      );
      return;
    }

    if (!(manualData[0].length == 3 || manualData[0].length == 4)) {
      $("#manual-data-error").html("please enter 2 or 3 dimensional data");
      return;
    }

    manualData.forEach((row, index) => {
      manualData[index] = row.map((cell, index) => {
        if (index == row.length - 1) {
          return cell.replace(/[\s+.*+?^${}()|[\]\\]/g, "");
        } else {
          return parseFloat(cell.replace(/[\s+*+?^${}()|[\]\\]/g, ""));
        }
      });
    });

    CURRENT_DATA = manualData;

    if (manualData[0].length === 3) {
      const transformedData = transform2dData(manualData);
      const parsedForChart = parse2dData(transformedData);
      createScatter(parsedForChart, "myChart");
    } else {
      const transformedData = transform3dData(manualData);
      const parsedForChart = parse3dData(transformedData);
      createScatter(parsedForChart, "myChart");
    }
  }
};

const getAverageDimensions = twoDArray => {
  let average = twoDArray[0].length;
  twoDArray.forEach(row => {
    average += row.length;
    average /= 2;
  });
  const finalAvg = Math.round(average);

  return finalAvg;
};

const parseManualData = () => {
  let final = [];
  const rawInput = $("#manual-data").val();

  const parsed = rawInput.split("\n");

  if (rawInput.includes(",")) {
    parsed.forEach((row, index) => {
      parsed[index] = row.split(",");
    });
    const dimensions = getAverageDimensions(parsed);

    final = parsed.filter(row => row.length == dimensions);
  } else {
    parsed.forEach((row, index) => {
      parsed[index] = row.split("\t");
    });
    const dimensions = getAverageDimensions(parsed);

    final = parsed.filter(row => row.length == dimensions);
  }

  return final;
};

const transform2dData = twoDArray => {
  const newDataObj = { result: [], test_data: [] };
  twoDArray.forEach((value, index) => {
    newDataObj.result.push(value[2]);
    newDataObj.test_data.push([value[0], value[1]]);
  });
  return newDataObj;
};

const transform3dData = twoDarray => {
  const newDataObj = { result: [], test_data: [] };
  twoDarray.forEach(value => {
    newDataObj.result.push(value[3]);
    newDataObj.test_data.push([value[0], value[1], value[2]]);
  });
  return newDataObj;
};

const initMethodChart = () => {
  createScatter([{}], "myChart");
  setTimeout(() => {
    $("#myChart-message").empty();
  }, 50);
};

initSampleData();
initMethodChart();
