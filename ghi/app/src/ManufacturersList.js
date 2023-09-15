import { useState, useEffect } from 'react';

function ManufacturersList() {

    // initialize variables and state 
    const [manufacturers, setManufacturers] = useState([]);

    // GET the list of manufacturers from the Inventory API
    async function loadManufacturers() {
        const response = await fetch('http://localhost:8100/api/manufacturers/');

        if (response.ok) {
            const data = await response.json();
            setManufacturers(data.manufacturers);
        } else {
            console.error(response);
        }
    };

    useEffect(() => {
        loadManufacturers();
    }, []);

    return (
        <div>
            <h1>Manufacturers</h1>
            <table className="table table-striped">
                <thead>
                    <tr>
                        <th>Name</th>
                    </tr>
                </thead>
                <tbody>
                    {manufacturers.map(manufacturer => {
                        return (
                            <tr key={manufacturer.id}>
                                <td>{manufacturer.name}</td>
                            </tr>
                        );
                    })}
                </tbody>
            </table>
        </div>
    );
}

export default ManufacturersList;