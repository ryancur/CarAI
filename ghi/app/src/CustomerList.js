import { useState, useEffect } from 'react';

function CustomerList() {

    const [customers, setCustomers] = useState([]);

    async function loadCustomers() {

        const response = await fetch('http://localhost:8090/api/customers/');

        if (response.ok) {
            const data = await response.json();
            setCustomers(data.customers);
        } else {
            console.error(response);
        }
    };


    const handleDelete = async (id) => {
        const deleteUrl = `http://localhost:8090/api/customers/${id}/`

        const fetchConfig = {
            method: "delete",
            headers: {
                'Content-Type': 'application/json',
            }
        }
        const response = await fetch(deleteUrl, fetchConfig)
        if (response.ok) {
            window.location.reload(false)
        }
    }

    useEffect(() => {
        loadCustomers();
    }, []);


    return (
        <div>
            <h1>Customers</h1>
            <table className="table table-striped">
                <thead>
                    <tr>
                        <th>First Name</th>
                        <th>Last Name</th>
                        <th>Phone Number</th>
                        <th>Address</th>
                    </tr>
                </thead>
                <tbody>
                    {customers.map(customer => {
                        return (
                            <tr key={customer.id}>
                                <td>{customer.first_name}</td>
                                <td>{customer.last_name}</td>
                                <td>{customer.phone_number}</td>
                                <td>{customer.address}</td>
                                <td><button onClick={() => handleDelete(customer.id)}>Delete</button></td>
                            </tr>
                        );
                    })}
                </tbody>
            </table>
        </div>
    );
}

export default CustomerList;
