// goToPage(SVMmethod, chartId, data.Id);
let myChartCreated = false;
let defaultAnimation = "easeInQuint";
let CURRENT_DATA = "Sample";

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
 * @param {float} pointConfidence confidence of for the data point label
 * @param {data point} data
 */
const getDataPointArgs = (classes, value, pointConfidence, data) => {
  const confidence = mapConfidence(pointConfidence);
  const newObjectClassIndex = getClassesIndex(classes, value);
  const dataPointArgs = {
    value: value,
    index: newObjectClassIndex,
    confidence: confidence,
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
      data_set["confidence"],
      data_set.test_data[index]
    );
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
      data_set["confidence"],
      data_set.test_data[index]
    );
    const dataPointObject = createDataPoint(dataPointArgs, 3);
    finalData.push(dataPointObject);
  });
  return finalData;
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
