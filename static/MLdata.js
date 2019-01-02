const printf = array_of_values => {
  array_of_values.forEach(value => {
    console.log(value);
  });
};

const showModel = () => {
  $.ajax({
    type: "GET",
    url: "/get_model/",
    contentType: "application/json; charset=utf-8",
    data: {
      myData: "this data"
    },
    success: function(data) {
      console.log(data);
    },
    error: function(jqXHR, textStatus, errorThrown) {
      alert(errorThrown);
    }
  });
};
var ctx = document.getElementById("myChart").getContext("2d");
var scatterChart = new Chart(ctx, {
  type: "scatter",
  data: {
    datasets: [
      {
        backgroundColor: "rgb(255, 99, 132)",
        borderColor: "rgb(255, 99, 132)",
        label: "Scatter Dataset",
        data: [
          {
            x: -10,
            y: 0
          },
          {
            x: 0,
            y: 10
          },
          {
            x: 10,
            y: 5
          },
          { x: 4, y: 3 },
          { x: 5, y: 6 }
        ]
      },
      {
        backgroundColor: "rgb(0, 0, 0)",
        borderColor: "rgb(0, 0, 0)",
        label: "Scatter Dataset2",
        data: [
          {
            x: -2,
            y: 5
          },
          {
            x: 5,
            y: 3
          },
          {
            x: 2,
            y: 4
          }
        ]
      }
    ]
  },

  options: {
    animation: {
      duration: 1000,
      easing: "linear"
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
// var scatterChart2 = new Chart(ctx, {
//   type: "scatter",

// });
// $.getJSON($SCRIPT_ROOT + "/get_model", {}, function(data) {
//   window.PageSettings = data;
//   $(idOfMin).append(data.minsize);
// });
