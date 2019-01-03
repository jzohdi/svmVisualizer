var waiting = false;
var myChartCreated = false;
var defaultAnimation = "easeInQuint";
var CURRENT_DATA = [];
const printf = array_of_values => {
  array_of_values.forEach(value => {
    console.log(value);
  });
};
const string = data => {
  return JSON.stringify(data);
};
const representData = [
  { color: "rgb(255, 99, 132)", label: "Classification 1" },
  { color: "rgb(0, 124, 249)", label: "Classification 2" }
];

const createNewDataObject = (newValue, newIndex) => {
  newObj = {};
  newObj.backgroundColor = representData[newIndex].color;
  newObj.borderColor = representData[newIndex].color;
  newObj.label = representData[newIndex].label;
  newObj.data = [];
  return newObj;
};

const parseData = data_set => {
  const classes = {};
  const finalData = [];
  data_set.result.forEach((value, index) => {
    if (!classes.hasOwnProperty(value)) {
      const newIndex = Object.keys(classes).length;
      newDataObject = createNewDataObject(value, newIndex);
      classes[value] = newIndex;
      const point = {
        x: data_set.test_data[index][0],
        y: data_set.test_data[index][1]
      };
      newDataObject.data.push(point);
      finalData.push(newDataObject);
    } else {
      const indexInFinalData = classes[value];
      const point = {
        x: data_set.test_data[index][0],
        y: data_set.test_data[index][1]
      };
      finalData[indexInFinalData].data.push(point);
    }
  });
  return finalData;
};
const cacheData = {};

const runMethod = chartId => {
  if (!waiting) {
    const selectedText = $("#select-method :selected").text(); // The text content of the selected option
    const selectedVal = $("#select-method").val();
    if (cacheData.hasOwnProperty(selectedVal)) {
      createScatter(cacheData[selectedVal].plot, chartId);
      $("#" + chartId + "-message").html(
        "Best Params: " + string(cacheData[selectedVal].params)
      );
    } else {
      showModel(selectedVal, chartId);
    }
  }
};
const showModel = (SVMmethod, chartId) => {
  waiting = true;
  $("#" + chartId + "-message").html("processing...");
  $.ajax({
    type: "GET",
    url: "/get_model/",
    contentType: "application/json; charset=utf-8",
    data: {
      runMethod: SVMmethod
    },
    success: function(data) {
      const bestParams = data["params"];
      delete data["params"];
      const bestScore = data["score"];
      delete data["score"];

      dataSet = parseData(data);

      cacheData[SVMmethod] = {
        plot: dataSet,
        score: bestScore,
        params: bestParams
      };

      createScatter(dataSet, chartId);
      waiting = false;
      console.log("method call successful");
      $("#" + chartId + "-message").html("Best Params: " + string(bestParams));
    },
    error: function(jqXHR, textStatus, errorThrown) {
      alert(errorThrown);
    }
  });
};
// let progress = { value: 0 };
const charts = {};
const createScatter = (dataSet, targetCanvas) => {
  if (charts.hasOwnProperty(targetCanvas)) {
    const thisScatterChart = charts[targetCanvas];
    thisScatterChart.data.datasets = dataSet;
    thisScatterChart.update();
  } else {
    const ctx = document.getElementById(targetCanvas).getContext("2d");
    const scatterChart = new Chart(ctx, {
      type: "scatter",
      data: {
        datasets: dataSet
      },

      options: {
        animation: {
          duration: 1000,
          easing: defaultAnimation
        },
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          xAxes: [
            {
              type: "linear",
              position: "bottom"
            }
          ],
          yAxes: [
            {
              ticks: {
                beginAtZero: true
              }
            }
          ]
        }
      }
    });
    charts[targetCanvas] = scatterChart;
  }
};

const runData = () => {
  const selectedData = $("#select-data :selected").text();
  if (selectedData === "Sample Data") {
    const sampleData = cacheData["Sample Data"].plot;
    createScatter(sampleData, "sampleData");
  } else if (selectedData === "Manual Data") {
    const manualData = parseManualData();

    manualData.forEach((row, index) => {
      manualData[index] = row.map(cell => {
        return parseFloat(cell);
      });
    });
  }
};

const parseManualData = () => {
  let final = [];
  const rawInput = $("#manual-data").val();
  if (rawInput.includes(",")) {
    const parsed = rawInput.split("\n");
    parsed.forEach((row, index) => {
      parsed[index] = row.split(",");
    });
    final = parsed.filter(row => row.length > 2);
  } else {
    const parsed = rawInput.split("\n");

    parsed.forEach((row, index) => {
      parsed[index] = row.split("\t");
    });
    final = parsed.filter(row => row.length > 2);
  }
  return final;
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
