const printf = array_of_values => {
  array_of_values.forEach(value => {
    console.log(value);
  });
};

const showModel = () => {
  $.ajax({
    type: "GET",
    url: "~/sklearn.py",
    contentType: "application/json; charset=utf-8",
    data: {
      myData: "this data"
    },
    success: function(data) {
      printf([data]);
    },
    error: function(jqXHR, textStatus, errorThrown) {
      alert(errorThrown);
    }
  });
};

// $.getJSON($SCRIPT_ROOT + "/get_model", {}, function(data) {
//   window.PageSettings = data;
//   $(idOfMin).append(data.minsize);
// });
