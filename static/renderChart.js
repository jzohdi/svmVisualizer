let waiting = false;
let myChartCreated = false;
let defaultAnimation = "easeInQuint";
let CURRENT_DATA = "Sample";
// TODO calculate these values based on the width of the chart and the number of data points.
const sizeFor3dPoint = 10;
const sizeFor2dPoint = 13;

const getChartWidth = () => {
  if (window.innerWidth < 1200) {
    return window.innerWidth * 0.85;
  } else {
    return 1200;
  }
  // return Math.min(window.innerWidth, window.innerHeight);
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
  width: 800, // default values for width and height
  height: 800,
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
  const dimensions = getChartWidth();
  const target = document.getElementById(targetCanvas); 
  layout.width = dimensions;
  layout.height = dimensions;
  // layout["margin-left"] = dimensions["margin-left"];
  if (target) {
    target.innerHTML = ""
    Plotly.newPlot(targetCanvas, dataSet, layout);
  } else {
    console.error("createScatter: no target canvas found")
  }
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

const mapConfidence = (confidenceSet, index, minScale) => {
  if (confidenceSet == null) {
    return false;
  }
  return mapData(confidenceSet[index], 0.49, 1, minScale, 0.99);
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
    const confidence = mapConfidence(data_set["confidence"], index, 0);
    const formattedRGBA = color.replace(
      "alpha",
      confidence.toFixed(6).toString()
    );
    trace.x.push(data_set.test_data[index][0]);
    trace.y.push(data_set.test_data[index][1]);
    trace.text.push("confidence: " + confidence.toFixed(8));
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
    const confidence = mapConfidence(data_set["confidence"], index, 0.4);
    const formattedRGBA = color.replace(
      "alpha",
      confidence.toFixed(3).toString()
    );
    trace.x.push(data_set.test_data[index][0]);
    trace.y.push(data_set.test_data[index][1]);
    trace.z.push(data_set.test_data[index][2]);
    trace.text.push("confidence: " + confidence.toFixed(8));
    trace.marker.color.push(formattedRGBA);
  });

  return Object.values(traces);
};

function parseModelData(data, chartId) {
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

  createScatter(dataSet, chartId);
  waiting = false;
  console.log("method call successful");
  document.getElementById(`${chartId}-message`).innerText = `Best Params: ${string(bestParams)}\n\nBest Score: ${bestScore}`;
};

function retrieveModel(SVMmethod, chartId, model_id) {
  // Do retrive model once at beginning to check if the data set is cached in data base.
  fetch("/retrieve_model/" + model_id).then(res => res.json()).then(data => {
    if (data["status"] === "Finished") {
      parseModelData(data, chartId);
    } else if (data.hasOwnProperty("error")) {
      clearInterval(intervalID);
    } else {
      retryRetrieve(chartId, model_id);
    }
  })
};

function retryRetrieve(chartId, model_id) {
  let intervalID = null;
  intervalID = setInterval(function() {
    fetch("/retrieve_model/" + model_id).then(res => res.json()).then(data => {
      if (data["status"] === "Finished") {
        parseModelData(data, chartId);
        clearInterval(intervalID);
      } else if (data.hasOwnProperty("error")) {
        clearInterval(intervalID);
      }
    });
  }, 5000);
}

const handleError = () => {
  const messageDiv = document.getElementById("myChart-message");
  if (messageDiv) {
    messageDiv.innerText = "URL not formed correctly...\n Going back to main page."
  }
  setTimeout(() => {
    location.href = "/svm_visualizer";
  }, 2000);
};

const makePlot = chartParams => {
  const params = JSON.parse(chartParams);
  if (
    params["method"] == null ||
    params["chart"] == null ||
    params["data"] == null
  ) {
    return handleError();
  }
  retrieveModel(params["method"], params["chart"], params["data"]);
};
// get the url query string
const queryString = window.location.search;
const urlParams = new URLSearchParams(queryString);

const chartParams = urlParams.get("params");

if (chartParams != null) {
  try {
    makePlot(chartParams);
  } catch (error) {
    handleError();
  }
} else {
  handleError();
}
