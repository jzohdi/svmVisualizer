        const modelIdInput = document.getElementById('polling-id')
        const plotlyTargetId = "myChart";
        const chartWrapper = document.getElementById(plotlyTargetId);
        const defaultAnimation = "easeInQuint";
        const representData = [
            { color: "rgba(255, 99, 132, alpha)", label: "Classification 1" },
            { color: "rgba(0, 124, 249, alpha)", label: "Classification 2" },
            { color: "rgba(25, 150, 100, alpha)", label: "Classification 3" },
            { color: "rgba(93, 1, 150, alpha)", label: "Classification 4" }
        ];
        const sizeFor3dPoint = 10;
        const sizeFor2dPoint = 12;

        const randomRGBA = _ => {
            var o = Math.round,
                r = Math.random,
                s = 255;
            return "rgba(" + o(r() * s) + "," + o(r() * s) + "," + o(r() * s) + ",alpha)";
        };
        /**
         * The color of the chart
         */ 
        const getNextColor = nextIndex => {
            if (nextIndex < representData.length) {
                return representData[nextIndex].color;
            }
            return randomRGBA();
        };

        function getNewBaseTrace(label, dimensionSize) {
            if (dimensionSize === 2) {
                return {
                    x: [],
                    y: [],
                    mode: "markers",
                    name: label,
                    text: [],
                    marker: {
                        size: sizeFor2dPoint,
                        opacity: [],
                        color: []
                    },
                    type: "scatter"
                };                
            }
            if (dimensionSize === 3) {
                return {
                    x: [],
                    y: [],
                    z: [],
                    mode: "markers",
                    name: label,
                    text: [],
                    marker: {
                        size: sizeFor3dPoint,
                        color: [],
                        opacity: [],
                        symbol: "circle"
                    },
                    type: "scatter3d"
                };
            }
            console.log("1 dimension data not supported yet");
            throw new Error("not supported graphing")
        }

        /**
         * Takes an array of labels: ["yes", "no", ...] or [1, 1, 0, etc.] 
         * and returns {
         *  "yes": plotlyTraceConfig,
         * "no": plotlyTraceConfig
         * }
         * The reason for doing this instead of using a color dimension
         * https://plotly.com/javascript/line-and-scatter/#scatter-plot-with-a-color-dimension
         * is that I don't find a good way to add opacity per circle (which will be based on confidence)
         */
        function makePlotlyTraces(labels, dimensionSize) {
            const traces = {}
            for (const label of labels) {
                if (traces[label] === undefined) {
                    traces[label] = getNewBaseTrace(label, dimensionSize)
                }
            }
            return traces;
        }
        function getColorForValue (usedColorsDict, value) {
            if (usedColorsDict.hasOwnProperty(value)) {
                return usedColorsDict[value];
            }
            usedColorsDict[value] = getNextColor(Object.keys(usedColorsDict).length);
            return usedColorsDict[value];
        };
        const mapData = (num, min_One, max_One, min_Out, max_Out) => {
            return (
                ((num - min_One) * (max_Out - min_Out)) / (max_One - min_One) + min_Out
            );
        };

        const mapConfidence = (confidence) => {
            return mapData(confidence, 0.49, 1, 0.5, 0.99);
        };
        /**
         * traces = { label1: traceConfig, label2: traceConfig, ... }
         * coordinates = [[x, y, z], [x, y, z], ...] (1-3 Dim)
         * labels = [label1, label2, label2, label2, label1]
         * confidences = [0.122222, 0.543111, ...] 
         * the length of coordinates, labels and confidences must all be equal
         */ 
        function fillTracesWithData(traces, coordinates, labels, confidences) {
            const labelColors = {}
            for (let index = 0; index < labels.length; index++) {
                const label = labels[index]; // get the label
                const plotlyConfig = traces[label] // get the plotly config for this label
                const confidence = mapConfidence(confidences[index]);
                plotlyConfig.marker.opacity.push(confidence)
                const color = getColorForValue(labelColors, label).replace("alpha", confidence);
                plotlyConfig.text.push(`value: ${label}`)
                plotlyConfig.marker.color.push(color);
                const coordinate = coordinates[index];
                if (coordinate[0] !== undefined) {
                    plotlyConfig.x.push(coordinate[0])
                } 
                if (coordinate[1] !== undefined) {
                    plotlyConfig.y.push(coordinate[1])
                }
                if (coordinate[2] !== undefined) {
                    plotlyConfig.z.push(coordinate[2])
                } else {
                    // plotlyConfig.z.push(0)
                }
            }
        }
        const layout = {
            title: "",
            autosize: true,
            hovermode: "closest",
            showlegend: false,
            width: 0,
            height: 0,
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
        function renderChart(data) {
            data = data.result;
            const plotCoordinates = JSON.parse(data.testing_data);
            if (plotCoordinates.length < 1) {
                return;
            }
            const labels = JSON.parse(data.result)
            const confidence = JSON.parse(data.confidence);
            // the number of dimensions of the the training data. ex: [[1, 2, 4], [0, 0, 1]] = 3
            // max dimensions that can be represented is 3
            const dimensionSize = Math.min(3, plotCoordinates[0].length); 
            const traces = makePlotlyTraces(labels, dimensionSize)
            fillTracesWithData(traces, plotCoordinates, labels, confidence);
            const plots = Object.values(traces);
            const dimension = Math.min(chartWrapper.getBoundingClientRect().width, 800);
            layout.width = dimension;
            layout.height = dimension;
            chartWrapper.innerHTML = ""
            Plotly.newPlot(plotlyTargetId, plots, layout, { resonsive: true });
        }
        
        let currentPollingInterval;
        
        const maxBackoff = 30000; // Cap the backoff at 30 seconds
        async function startPollingFromId(backoffMs) {
            chartWrapper.innerHTML = `<div class="sk-cube-grid">
            <div class="sk-cube sk-cube1"></div>
            <div class="sk-cube sk-cube2"></div>
            <div class="sk-cube sk-cube3"></div>
            <div class="sk-cube sk-cube4"></div>
            <div class="sk-cube sk-cube5"></div>
            <div class="sk-cube sk-cube6"></div>
            <div class="sk-cube sk-cube7"></div>
            <div class="sk-cube sk-cube8"></div>
            <div class="sk-cube sk-cube9"></div>
          </div></div>`
            const id = modelIdInput.value.trim();
            if (!id) {
                alert('Please enter a valid ID.');
                return;
            }

            // Clear any previous polling interval
            if (currentPollingInterval) {
                clearInterval(currentPollingInterval);
            }
            currentPollingInterval = setTimeout(async () => {
                try {
                    const response = await fetch(`/api/train_result/${id}`);
                    if (!response.ok) {
                        throw new Error(`HTTP error! Status: ${response.status}`);
                    }

                    const data = await response.json();

                    if (data.status === 'complete') {
                        clearInterval(currentPollingInterval);
                        currentPollingInterval = null;
                        console.log('Data is complete. Rendering chart...');
                        // Trigger chart rendering here
                        renderChart(data); // Replace this with your chart rendering logic
                    } else if(data.status === "pending") {
                        console.log('Status is pending, will try again in 1000ms');
                        startPollingFromId(Math.min(backoffMs * 2, maxBackoff))
                    }
                } catch (error) {
                    console.error('Error during polling:', error);
                    clearInterval(currentPollingInterval); // Stop polling on error
                    chartWrapper.innerHTML = "Error..."
                }
            }, backoffMs); // Poll every 1 second
        }


        const table = document.getElementById('dynamic-table');
        const headerRow = document.getElementById('table-header');
        const addColumnButton = document.getElementById('add-column');
        const removeColumnButton = document.getElementById('remove-column');
        const addRowButton = document.getElementById('add-row');
        const removeRowButton = document.getElementById('remove-row');
        const importCsvInput = document.getElementById('import-csv');
        const exportCsvButton = document.getElementById('export-csv');
        const submitButton = document.getElementById("submit");
        const loadSampleButton = document.getElementById('load-sample');
        
        async function loadSampleDataSet(sampleName) {
            try {
                const response = await fetch(`/static/${sampleName}`);
                if (!response.ok) {
                    throw new Error(`Failed to load dataset: ${response.status}`);
                }
                const csvContent = await response.text();
                console.log(csvContent)
                // Parse CSV content and load it into the table
                const lines = csvContent.split('\n');
                loadLines(lines)

                console.log(`Loaded dataset: ${sampleName}`);
            } catch (error) {
                console.error('Error loading sample dataset:', error);
                alert('Failed to load the sample dataset.');
            }
        }

        loadSampleButton.addEventListener('click', async () => {
            const dataset = document.getElementById('sample-dataset').value;
            loadSampleDataSet(dataset)
        });

        // Add a new column
        addColumnButton.addEventListener('click', () => {
            const columnCount = headerRow.children.length;
            const newColumnNumber = columnCount; // Column numbers start from 1

            // Add header cell
            const newHeaderCell = document.createElement('th');
            newHeaderCell.className = 'border border-gray-400 px-4 py-2';
            newHeaderCell.textContent = newColumnNumber;
            headerRow.appendChild(newHeaderCell);

            // Add a cell to each existing row
            Array.from(table.tBodies[0].rows).forEach(row => {
                const newCell = document.createElement('td');
                newCell.setAttribute('contenteditable', 'true');
                newCell.className = 'border border-gray-400 px-4 py-2';
                row.appendChild(newCell);
            });
        });

        // Remove the last column
        removeColumnButton.addEventListener('click', () => {
            const columnCount = headerRow.children.length;

            // Ensure we have at least "Label" and one numbered column
            if (columnCount > 2) {
                // Remove the last header cell
                headerRow.removeChild(headerRow.lastElementChild);

                // Remove the last cell from each row
                Array.from(table.tBodies[0].rows).forEach(row => {
                    row.removeChild(row.lastElementChild);
                });
            } else {
                alert('Cannot remove more columns. The table must include "Label" and at least one numbered column.');
            }
        });

        // Add a new row
        addRowButton.addEventListener('click', () => {
            const columnCount = headerRow.children.length;
            const newRow = document.createElement('tr');

            // Add a "Label" cell
            const labelCell = document.createElement('td');
            labelCell.className = 'border border-gray-400 px-4 py-2';
            labelCell.textContent = `Row ${table.tBodies[0].rows.length + 1}`;
            newRow.appendChild(labelCell);

            // Add empty cells for each numbered column
            for (let i = 1; i < columnCount; i++) {
                const newCell = document.createElement('td');
                newCell.setAttribute('contenteditable', 'true');
                newCell.className = 'border border-gray-400 px-4 py-2';
                newRow.appendChild(newCell);
            }

            table.tBodies[0].appendChild(newRow);
        });

        // Remove the last row
        removeRowButton.addEventListener('click', () => {
            const rowCount = table.tBodies[0].rows.length;

            // Ensure at least one row remains
            if (rowCount > 1) {
                table.tBodies[0].removeChild(table.tBodies[0].lastElementChild);
            } else {
                alert('Cannot remove more rows. The table must have at least one row.');
            }
        });

        function clearTbodyRows() {
            const tbody = table.tBodies[0];
            while (tbody.firstChild) {
                tbody.removeChild(tbody.firstChild);
            }
            const headerRow = document.getElementById("table-header");
            while (headerRow.children.length > 2) {
               headerRow.removeChild(headerRow.lastElementChild);
            }
        }

        function loadLines(lines) {
            clearTbodyRows();
            lines.forEach(line => {
                    const values = line.split(',');
                    if (values.length > 0 && values[0].trim() !== "") {
                        const newRow = document.createElement('tr');

                        values.forEach((value, index) => {
                            // Add columns dynamically if necessary
                            if (headerRow.children.length <= index) {
                                const newHeaderCell = document.createElement('th');
                                newHeaderCell.className = 'border border-gray-400 px-4 py-2';
                                newHeaderCell.textContent = index === 0 ? 'Label' : index;
                                headerRow.appendChild(newHeaderCell);
                            }

                            const cell = document.createElement('td');
                            cell.setAttribute('contenteditable', 'true');
                            cell.className = 'border border-gray-400 px-4 py-2';
                            cell.textContent = value.trim();
                            newRow.appendChild(cell);
                        });

                        table.tBodies[0].appendChild(newRow);
                    }
                });
        }

        // Import CSV
        importCsvInput.addEventListener('change', (event) => {
            const file = event.target.files[0];
            if (!file) return;

            clearTbodyRows();
            const reader = new FileReader();
            reader.onload = (e) => {
                const lines = e.target.result.split('\n');
                loadLines(lines)
            };

            reader.readAsText(file);
        });

        // Export CSV
        exportCsvButton.addEventListener('click', () => {
            const rows = Array.from(table.tBodies[0].rows);
            const csvContent = rows.map(row => {
                return Array.from(row.children).map(cell => cell.textContent).join(',');
            }).join('\n');

            const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
            const url = URL.createObjectURL(blob);
            const link = document.createElement('a');
            link.href = url;
            link.download = 'table_data.csv';
            link.click();
            URL.revokeObjectURL(url);
        });

        const parseTableData = () => {
            const rows = Array.from(table.tBodies[0].rows);
            const labels = rows.map(row => row.cells[0].textContent.trim());
            const trainingData = rows.map(row => {
                return Array.from(row.cells).slice(1).map(cell => parseFloat(cell.textContent.trim()) || 0);
            });
            const method = document.getElementById('method').value;

            return {
                training_data: trainingData,
                labels: labels,
                method: method
            };
        };

        submitButton.addEventListener("click", async () => {
            const payload = parseTableData();
            const response = await fetch("/api/train_model", 
                {
                    method: "POST",
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify(payload)
                }
            );
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const result = await response.json();
            if (["complete", "pending"].includes(result.status)) {
                modelIdInput.value = result.id;
                startPollingFromId(1000)
            }
        })

        async function showExample() {
            const loadSample = await loadSampleDataSet("example2.csv");
            submitButton.click();
        }
        showExample();
