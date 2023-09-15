import React, { useState, useEffect } from "react";
import { Bar, Doughnut } from "react-chartjs-2";
import axios from "axios";
import { Chart, registerables } from "chart.js";
Chart.register(...registerables);

const Dashboard = () => {
	const [salesData, setSalesData] = useState([]);
	const [manufacturerData, setManufacturerData] = useState([]);
	const [loading, setLoading] = useState(true);

	const fetchData = async () => {
		const salesResponse = await axios.get("http://localhost:8090/api/sales/");

		let salesbysalesperson = [];
		let salescount = {};


		for (let sale of salesResponse.data.sales) {
			let fullName = sale.salesperson.first_name + " " + sale.salesperson.last_name;
			// salesbysalesperson.push(sale.salesperson.employee_id);
			salesbysalesperson.push(fullName);
		}

		for (let sale of salesbysalesperson) {
			if (!(sale in salescount)) {
				salescount[sale] = 1;
			} else {
				salescount[sale] += 1;
			}
		}

		let salesbymanufacturer = []
		let manufacturercount = {}

		for (let manufacturer of salesResponse.data.sales) {
			salesbymanufacturer.push(manufacturer.automobile.manufacturer_name);
		}

		for (let manufacturer of salesbymanufacturer) {
			if (!(manufacturer in manufacturercount)) {
				manufacturercount[manufacturer] = 1;
			} else {
				manufacturercount[manufacturer] += 1;
			}
		}

		setManufacturerData(manufacturercount);
		setSalesData(salescount);

		setLoading(false);
	};

	console.log(salesData);
	const salesChartData = salesData
		? {
			labels: Object.keys(salesData).map((id) => id),

			datasets: [
				{
					label: "Sales by Salesperson",
					data: Object.keys(salesData).map((id) => salesData[id]),
				},
			],
		}
		: null;

	const inventoryChartData = manufacturerData ? {
		labels: Object.keys(manufacturerData).map((name) => name),
		datasets: [
			{
				label: "Sales by Manufacturer",
				data: Object.keys(manufacturerData).map((name) => manufacturerData[name]),

			},
		],
	} : null;


	useEffect(() => {
		fetchData();
	}, []);

	if (loading) {
		return <div>Loading...</div>;
	}

	return (
		<div style={{ width: "800px", height: "600px", margin: "auto" }}>
			<h2>Sales by Salesperson</h2>
			<Bar data={salesChartData} />
			<h2>Automobiles Sold by Manufacturer</h2>
			<Doughnut data={inventoryChartData} />

		</div>
	);
};

export default Dashboard;
