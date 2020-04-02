function getChart(id, type="bar", labels=[]) {
	return new Chart($("#" + id), {type: type, data: {labels: lables, datasets: []}});
}

function addChartData(chart, label, data, type="bar") {
	chart.data.datasets.push({type: type, label: label, data: data});
	chart.update();
}

function removeChartData(chart) {
	chart.data.datasets.forEach((dataset) => {
		dataset.data.pop();
	});
	chart.update();
}