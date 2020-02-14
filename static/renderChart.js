let waiting = false;
let myChartCreated = false;
let defaultAnimation = "easeInQuint";
let CURRENT_DATA = "Sample";

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
  $("#" + targetCanvas).empty();
  Plotly.newPlot(targetCanvas, dataSet, layout, { resonsive: true });
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
const getMarkerObject = (size, color, confidence, symbol = false) => {
  const newObj = {
    size: size,
    color: color,
    opacity: confidence ? confidence : 1
  };
  if (symbol) {
    newObj["symbol"] = symbol;
  }
  return newObj;
};

const getRandomColor = () => {
  const letters = "0123456789ABCDEF";
  let color = "#";
  for (let i = 0; i < 6; i++) {
    color += letters[Math.floor(Math.random() * 16)];
  }
  return color;
};
const representData = [
  { color: "rgb(255, 99, 132)", label: "Classification 1" },
  { color: "rgb(0, 124, 249)", label: "Classification 2" },
  { color: "rgb(25, 150, 100)", label: "Classification 3" },
  { color: "rgb(93, 1, 150)", label: "Classification 4" }
];

const getColorForObject = index => {
  if (index < representData.length) {
    return representData[index].color;
  }
  return getRandomColor();
};

const newJsonObject = confidence => {
  const newObj = {};
  newObj.x = [];
  newObj.y = [];
  if (confidence) {
    newObj.text = ["confidence: " + confidence.toFixed(8)];
  }
  newObj.mode = "markers";
  return newObj;
};

const add3dSettings = (object, color, confidence) => {
  object.type = "scatter3d";
  object.z = [];
  object.marker = getMarkerObject(
    sizeFor3DPoint(),
    color,
    confidence,
    "circle"
  );
  return object;
};

const addSettingsToNewObj = (object, dimensions, color, confidence) => {
  if (dimensions === 2) {
    object.type = "scatter";
    object.marker = getMarkerObject(sizeFor2DPoint(), color, confidence);
    return object;
  }
  return add3dSettings(object, color, confidence);
};

const createNewDataObject = (newIndex, dimensions, confidence = false) => {
  /* if the index is less than the default color array length, choose
  the color from the array. Otherwise, get a random color.*/
  const color = getColorForObject(newIndex);
  const newObj = newJsonObject(confidence);

  return addSettingsToNewObj(newObj, dimensions, color, confidence);
};

const createDataPoint = (dataPointParams, dimensions) => {
  const data = dataPointParams.data;
  const newDataObject = createNewDataObject(
    dataPointParams.index,
    dimensions,
    dataPointParams.confidence
  );
  newDataObject.x.push(data[0]);
  newDataObject.y.push(data[1]);
  if (dimensions > 2) {
    newDataObject.z.push(data[2]);
  }
  return newDataObject;
};

const getClassesIndex = (classes, value) => {
  if (classes.hasOwnProperty(value)) {
    return classes[value];
  }
  classes[value] = Object.keys(classes).length;
  return classes[value];
};

const mapConfidence = (confidenceSet, index) => {
  if (confidenceSet == null) {
    return false;
  }
  return mapData(confidenceSet[index], 0.49, 1, 0, 0.9);
};

/**
 *
 * @param {object} classes
 * @param {label} value label for the data point clasified
 * @param {data point} data
 */
const getDataPointArgs = (classes, value, data) => {
  const newObjectClassIndex = getClassesIndex(classes, value);
  const dataPointArgs = {
    value: value,
    index: newObjectClassIndex,
    data: data
  };
  return dataPointArgs;
};

const parse2dData = data_set => {
  const classes = {};
  const finalData = [];

  data_set.result.forEach((value, index) => {
    const dataPointArgs = getDataPointArgs(
      classes,
      value,
      data_set.test_data[index]
    );
    if (data_set["confidence"] != null) {
      dataPointArgs["confidence"] = mapData(
        data_set["confidence"][index],
        0.49,
        1,
        0,
        1
      );
    }
    const dataPointObject = createDataPoint(dataPointArgs, 2);
    finalData.push(dataPointObject);
  });
  return finalData;
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
  const classes = {};
  const finalData = [];
  data_set.result.forEach((value, index) => {
    const dataPointArgs = getDataPointArgs(
      classes,
      value,
      data_set.test_data[index]
    );
    // console.log("confidence..", data_set["confidence"][index]);
    if (data_set["confidence"] != null) {
      dataPointArgs["confidence"] = mapData(
        data_set["confidence"][index],
        0.49,
        1,
        0,
        0.9
      );
    }
    const dataPointObject = createDataPoint(dataPointArgs, 3);
    finalData.push(dataPointObject);
  });
  return finalData;
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

  createScatter(dataSet, chartId);
  waiting = false;
  console.log("method call successful");
  $("#" + chartId + "-message").html("Best Params: " + string(bestParams));
};

const retrieveModel = (SVMmethod, chartId, model_id) => {
  // Do retrive model once at beginning to check if the data set is cached in data base.
  $.get("/retrieve_model/" + model_id).done(function(data) {
    if (data["status"] === "Finished") {
      parseModelData(data, SVMmethod, chartId);
    } else if (data.hasOwnProperty("error")) {
      clearInterval(intervalID);
    } else {
      let intervalID = null;
      intervalID = setInterval(function() {
        $.get("/retrieve_model/" + model_id).done(function(data) {
          if (data["status"] === "Finished") {
            parseModelData(data, SVMmethod, chartId);
            clearInterval(intervalID);
          } else if (data.hasOwnProperty("error")) {
            clearInterval(intervalID);
          }
        });
      }, 5000);
    }
  });
};

const handleError = () => {
  $("#myChart-message").html(
    "URL not formed correctly...\n Going back to main page."
  );
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
