import { useState, useEffect } from "react";

function SalespeopleList() {
  const [salespeople, setSalespeople] = useState([]);

  async function loadSalespeople() {
    const response = await fetch("http://localhost:8090/api/salespeople/");

    if (response.ok) {
      const data = await response.json();
      setSalespeople(data.salespeople);
    } else {
      console.error(response);
    }
  }

  const handleDelete = async (employee_id) => {
    const deleteUrl = `http://localhost:8090/api/salespeople/${employee_id}/`;

    const fetchConfig = {
      method: "delete",
      headers: {
        "Content-Type": "application/json",
      },
    };
    const response = await fetch(deleteUrl, fetchConfig);
    if (response.ok) {
      window.location.reload(false);
    }
  };

  useEffect(() => {
    loadSalespeople();
  }, []);

    return (
        <div>
            <h1>Salespeople</h1>
            <table className="table table-striped">
                <thead>
                    <tr>
                        <th>Employee ID</th>
                        <th>First Name</th>
                        <th>Last Name</th>
                    </tr>
                </thead>
                <tbody>
                    {salespeople.map((salesperson) => {
                        return (
                            <tr key={salesperson.employee_id}>
                                <td>{salesperson.employee_id}</td>
                                <td>{salesperson.first_name}</td>
                                <td>{salesperson.last_name}</td>
                                <td>
                                    <button onClick={() => handleDelete(salesperson.employee_id)}>
                                        Delete
                                    </button>
                                </td>
                            </tr>
                        );
                    })}
                </tbody>
            </table>
        </div>
    );
}

export default SalespeopleList;
