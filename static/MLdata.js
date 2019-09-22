var waiting = false;
var myChartCreated = false;
var defaultAnimation = "easeInQuint";
var CURRENT_DATA = "Sample";

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

const representData = [
  { color: "rgb(255, 99, 132)", label: "Classification 1" },
  { color: "rgb(0, 124, 249)", label: "Classification 2" },
  { color: "rgb(25, 150, 100)", label: "Classification 3" },
  { color: "rgb(93, 1, 150)", label: "Classification 4" }
];

const createNewDataObject = (
  newValue,
  newIndex,
  dimensions,
  confidence = false
) => {
  newObj = {};

  newObj.x = [];
  newObj.y = [];
  if (confidence) {
    newObj.text = ["confidence: " + confidence.toFixed(8)];
  }
  newObj.mode = "markers";
  if (dimensions === 2) {
    newObj.type = "scatter";
    newObj.marker = getMarkerObject(
      10,
      representData[newIndex].color,
      confidence
    );
    // newObj.name = representData[newIndex].label;
    return newObj;
  } else {
    newObj.type = "scatter3d";
    newObj.z = [];
    newObj.marker = getMarkerObject(
      20,
      representData[newIndex].color,
      confidence,
      "circle"
    );
    // newObj.name = representData[newIndex].label;
    return newObj;
  }
};

const parse2dData = data_set => {
  const classes = {};
  const finalData = [];

  data_set.result.forEach((value, index) => {
    let confidence = false;

    if (data_set["confidence"] != null) {
      confidence = mapData(data_set["confidence"][index], 0.5, 1, 0, 1);
    }

    if (!classes.hasOwnProperty(value)) {
      const newIndex = Object.keys(classes).length;
      const newDataObject = createNewDataObject(value, newIndex, 2, confidence);
      classes[value] = newIndex;

      newDataObject.x.push(data_set.test_data[index][0]);
      newDataObject.y.push(data_set.test_data[index][1]);

      finalData.push(newDataObject);
    } else {
      const finalIndex = classes[value];
      const newDataObject = createNewDataObject(
        value,
        finalIndex,
        2,
        confidence
      );
      newDataObject.x.push(data_set.test_data[index][0]);
      newDataObject.y.push(data_set.test_data[index][1]);
      finalData.push(newDataObject);
      // finalData[finalIndex].x.push(data_set.test_data[index][0]);
      // finalData[finalIndex].y.push(data_set.test_data[index][1]);
    }
  });
  // console.log(finalData);
  return finalData;
};

const parse3dData = data_set => {
  const classes = {};
  const finalData = [];
  data_set.result.forEach((value, index) => {
    let confidence = false;
    if (data_set["confidence"] != null) {
      confidence = mapData(data_set["confidence"][index], 0.5, 1, 0, 0.9);
    }

    if (!classes.hasOwnProperty(value)) {
      const newIndex = Object.keys(classes).length;
      newDataObject = createNewDataObject(value, newIndex, 3, confidence);
      classes[value] = newIndex;

      newDataObject.x.push(data_set.test_data[index][0]);
      newDataObject.y.push(data_set.test_data[index][1]);
      newDataObject.z.push(data_set.test_data[index][2]);

      finalData.push(newDataObject);
    } else {
      const finalIndex = classes[value];
      newDataObject = createNewDataObject(value, finalIndex, 3, confidence);
      newDataObject.x.push(data_set.test_data[index][0]);
      newDataObject.y.push(data_set.test_data[index][1]);
      newDataObject.z.push(data_set.test_data[index][2]);
      finalData.push(newDataObject);
    }
  });
  return finalData;
};
const cacheData = {};

const runMethod = chartId => {
  if (!waiting) {
    const selectedText = $("#select-method :selected").text(); // The text content of the selected option
    const selectedVal = $("#select-method").val();
    if (cacheData.hasOwnProperty(selectedVal) && CURRENT_DATA === "Sample") {
      createScatter(cacheData[selectedVal].plot, chartId);
      $("#" + chartId + "-message").html(
        "Best Params: " + string(cacheData[selectedVal].params)
      );
    } else {
      showModel(selectedVal, chartId);
    }
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
  waiting = false;
  console.log("method call successful");
  $("#" + chartId + "-message").html("Best Params: " + string(bestParams));
};

const retrieveModel = (SVMmethod, chartId, model_id) => {
  // console.log(SVMmethod);
  // console.log(model_id);
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
          console.log(data);
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

const showModel = (SVMmethod, chartId) => {
  // console.log(CURRENT_DATA);
  waiting = true;
  $("#" + chartId + "-message").html("processing...");
  $.post("/train_model/", {
    runMethod: SVMmethod,
    data_set: CURRENT_DATA === "Sample" ? CURRENT_DATA : string(CURRENT_DATA)
  })
    .done(function(data) {
      retrieveModel(SVMmethod, chartId, data);
    })
    .fail(function(error) {
      alert(error);
    });
};
const getChartWidth = () => {
  if (window.innerWidth < 1200) {
    return window.innerWidth * 0.7;
  } else {
    return window.innerWidth / 2.3;
  }
};
// const calculateMargin = () => {
//   if (window.innerWidth < 2000) {
//     return window.innerWidth;
//   } else {
//     return window.innerWidth / 2;
//   }
// };
// let progress = { value: 0 };
const calculateDimension = () => {
  return { d: getChartWidth() };
};
const charts = {};
const layout = {
  title: "",
  autosize: true,
  hovermode: "closest",
  showlegend: false,
  width: 600,
  height: 600,
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
    createScatter(sampleData, "sampleData");
  } else if (selectedData === "Manual Data") {
    const manualData = parseManualData();
    if (manualData[0] == undefined) {
      $("#manual-data-error").html(
        "Could not parse data, check that all rows are filled appropriately."
      );
      return;
    }
    if (!(manualData[0].length == 3 || manualData[0].length == 4)) {
      console.log("here");
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
      createScatter(parsedForChart, "sampleData");
    } else {
      const transformedData = transform3dData(manualData);
      const parsedForChart = parse3dData(transformedData);
      createScatter(parsedForChart, "sampleData");
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
// const sampleData = [
//   {
//     backgroundColor: "rgb(255, 0, 0)",
//     borderColor: "rgb(255, 0, 0)",
//     label: "scatter1",
//     data: [{ x: 10, y: 2 }, { x: 2, y: 5 }, { x: 1, y: 4 }]
//   }
// ];
// createScatter(sampleData, "myChart");

const initSampleData = () => {
  showModel("Sample Data", "sampleData");
};

const initMethodChart = () => {
  createScatter([{}], "myChart");
  setTimeout(() => {
    $("#myChart-message").empty();
  }, 50);
};

initSampleData();
initMethodChart();
// var scatterChart2 = new Chart(ctx, {
//   type: "scatter",

// });
// $.getJSON($SCRIPT_ROOT + "/get_model", {}, function(data) {
//   window.PageSettings = data;
//   $(idOfMin).append(data.minsize);
// });
