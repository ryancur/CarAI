import { useState, useEffect } from 'react';

function SalespersonHistory() {

    const [sales, setSales] = useState([]);

    const [salespeople, setSalespeople] = useState([]);

    const [selectedSalesperson, setSelectedSalesperson] = useState('');

    async function loadSales() {

        const response = await fetch('http://localhost:8090/api/sales/');

        if (response.ok) {
            const data = await response.json();
            setSales(data.sales);
        } else {
            console.error(response);
        }
    };

    async function loadSalespeople() {

        const response = await fetch('http://localhost:8090/api/salespeople/');

        if (response.ok) {
            const data = await response.json();
            setSalespeople(data.salespeople);
        } else {
            console.error(response);
        }
    };

    const handleSalespersonChange = (event) => {
        const value = Number(event.target.value);
        setSelectedSalesperson(value);
    }

    useEffect(() => {
        loadSales();
        loadSalespeople();
    }, []);

    return (
        <div>
            <h1>Salesperson History</h1>
            <select onChange={handleSalespersonChange} value={selectedSalesperson} required name="salesperson" id="salesperson" className="form-select">
                <option value="">
                    Choose a salesperson
                </option>
                {salespeople.map(salesperson => {
                    return (
                        <option key={salesperson.employee_id} value={salesperson.employee_id}>
                            {salesperson.first_name} {salesperson.last_name}
                        </option>
                    )
                })}
            </select>
            <table className="table table-striped">
                <thead>
                    <tr>
                        <th>Salesperson</th>
                        <th>Customer</th>
                        <th>VIN</th>
                        <th>Price</th>
                    </tr>
                </thead>
                <tbody>
                    {sales.filter((sale) => sale.salesperson.employee_id === selectedSalesperson).map(sale => {
                        return (
                            <tr key={sale.id}>
                                <td>{sale.salesperson.first_name} {sale.salesperson.last_name}</td>
                                <td>{sale.customer.first_name} {sale.customer.last_name}</td>
                                <td>{sale.automobile.vin}</td>
                                <td>${sale.price}</td>
                            </tr>
                        );
                    })}
                </tbody>
            </table>
        </div>
    );
}

export default SalespersonHistory;
