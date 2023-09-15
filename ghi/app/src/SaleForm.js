import React, { useEffect, useState } from 'react';

function SaleForm() {
    const [automobiles, setAutomobiles] = useState([]);

    const [salesperson, setSalesperson] = useState([]);

    const [customers, setCustomers] = useState([]);

    const [price, setPrice] = useState('');

    const [selectedSalesperson, setSelectedSalesperson] = useState('');

    const [selectedCustomer, setSelectedCustomer] = useState('');

    const [selectedAutomobile, setSelectedAutomobile] = useState('');

    const handlePriceChange = (event) => {
        const value = event.target.value;
        setPrice(value);
    }

    const handleSelectedSalespersonChange = (event) => {
        const value = event.target.value;
        setSelectedSalesperson(value);
    }

    const handleSelectedCustomerChange = (event) => {
        const value = event.target.value;
        setSelectedCustomer(value);
    }

    const handleSelectedAutomobileChange = (event) => {
        const value = event.target.value;
        setSelectedAutomobile(value);
    }

    const handleSubmit = async (event) => {
        event.preventDefault();
        const data = {};
        const putData = {};

        data.automobile = selectedAutomobile;
        data.salesperson = selectedSalesperson;
        data.customer = selectedCustomer;
        data.price = price;


        const saleUrl = 'http://localhost:8090/api/sales/';
        const fetchConfig = {
            method: "post",
            body: JSON.stringify(data),
            headers: {
                'Content-Type': 'application/json',
            }
        }
        const response = await fetch(saleUrl, fetchConfig);
        if (response.ok) {
            setSelectedSalesperson('');
            setSelectedCustomer('');
            setSelectedAutomobile('');
            setPrice('');
        }


        const automobileUrl = `http://localhost:8100/api/automobiles/${selectedAutomobile}/`;
        const automobileFetchConfig = {
            method: "put",
            body: JSON.stringify({ sold: true }),
            headers: {
                'Content-Type': 'application/json',
            }
        }
        const automobileResponse = await fetch(automobileUrl, automobileFetchConfig);
    };

    async function loadAutomobiles() {
        const automobilesUrl = 'http://localhost:8100/api/automobiles/';
        const response = await fetch(automobilesUrl);
        if (response.ok) {
            const data = await response.json();
            const availableAutomobiles = data.autos.filter((automobile) => !automobile.sold);
            setAutomobiles(availableAutomobiles);
        } else {
            console.error(response);
        }
    }

    async function loadSalespeople() {
        const salespeopleUrl = 'http://localhost:8090/api/salespeople/';
        const response = await fetch(salespeopleUrl);
        if (response.ok) {
            const data = await response.json();
            setSalesperson(data.salespeople);
        } else {
            console.error(response);
        }
    }

    async function loadCustomers() {
        const customersUrl = 'http://localhost:8090/api/customers/';
        const response = await fetch(customersUrl);
        if (response.ok) {
            const data = await response.json();
            setCustomers(data.customers);
        } else {
            console.error(response);
        }
    }

    useEffect(() => {
        loadAutomobiles();
        loadSalespeople();
        loadCustomers();
    }, []);

    return (
        <div className="row">
            <div className="offset-3 col-6">
                <div className="shadow p-4 mt-4">
                    <h1>Create a new sale</h1>
                    <form onSubmit={handleSubmit} id="create-sale-form">
                        <div className="mb-3">
                            <select onChange={handleSelectedAutomobileChange} value={selectedAutomobile} required name="automobile" id="automobile" className="form-select">
                                <option value="">
                                    Choose an automobile VIN
                                </option>
                                {automobiles.filter((automobile) => automobile.sold === false).map(automobile => {
                                    return (
                                        <option key={automobile.vin} value={automobile.vin}>
                                            {automobile.vin}
                                        </option>
                                    )
                                })}
                            </select>
                        </div>
                        <div className="mb-3">
                            <select onChange={handleSelectedSalespersonChange} value={selectedSalesperson} required name="salesperson" id="salesperson" className="form-select">
                                <option value="">
                                    Choose a salesperson
                                </option>
                                {salesperson.map(person => {
                                    return (
                                        <option key={person.employee_id} value={person.employee_id}>
                                            {person.first_name} {person.last_name}
                                        </option>
                                    )
                                })}
                            </select>
                        </div>
                        <div className="mb-3">
                            <select onChange={handleSelectedCustomerChange} value={selectedCustomer} required name="customer" id="customer" className="form-select">
                                <option value="">
                                    Choose a customer
                                </option>
                                {customers.map(customer => {
                                    return (
                                        <option key={customer.id} value={customer.id}>
                                            {customer.first_name} {customer.last_name}
                                        </option>
                                    )
                                })}
                            </select>
                        </div>
                        <div className="form-floating mb-3">
                            <input onChange={handlePriceChange} value={price} placeholder="Price"
                                required
                                type="number"
                                name="price"
                                id="price"
                                className="form-control" />
                            <label htmlFor="price">Price</label>
                        </div>
                        <button className="btn btn-primary">Create</button>
                    </form>
                </div>
            </div>
        </div>
    )
}

export default SaleForm
