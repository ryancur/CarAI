import { useState, useEffect } from 'react';

function TechnicianList() {

    // initialize variables and state
    const [technicians, setTechnicians] = useState([]);

    // GET the list of technicians from the service API
    async function loadTechnicians() {
        const response = await fetch('http://localhost:8080/api/technicians/');

        if (response.ok) {
            const data = await response.json();
            setTechnicians(data.technicians);
        } else {
            console.error(response);
        }
    };

    // DELETE a single technician using their id
    const handleDelete = async (id) => {
        const url = `http://localhost:8080/api/technicians/${id}/`
        const fetchConfig = {
            method: "delete",
            headers: {
                'Content-Type': 'application/json'
            },
        };
        const response = await fetch(url, fetchConfig);
        if (response.ok) {
            loadTechnicians();
        }
    };

    useEffect(() => {
        loadTechnicians();
    }, []);

    return (
        <div>
            <h1>Technicians</h1>
            <table className="table table-striped">
                <thead>
                    <tr>
                        <th>Employee ID</th>
                        <th>First Name</th>
                        <th>Last Name</th>
                    </tr>
                </thead>
                <tbody>
                    {technicians.map(tech => {
                        return (
                            <tr key={tech.id}>
                                <td>{tech.employee_id}</td>
                                <td>{tech.first_name}</td>
                                <td>{tech.last_name}</td>
                                <td><button onClick={() => handleDelete(tech.id)}>Delete</button></td>
                            </tr>
                        );
                    })}
                </tbody>
            </table>
        </div>
    );
}

export default TechnicianList;
