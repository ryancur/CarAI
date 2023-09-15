import { useState, useEffect } from "react";

function AppointmentsHistory() {
    const [appointments, setAppointments] = useState([]);
    const [searchInput, setSearchInput] = useState("");

    const handleSearchBarChange = (event) => {
        const value = event.target.value;
        setSearchInput(value);
    };

    const handleSubmit = async (event) => {
        event.preventDefault();
        loadAppointments();
    };

    async function loadAppointments() {
        const response = await fetch("http://localhost:8080/api/appointments/");

        if (response.ok) {
            const data = await response.json();

            const newArray = data.appointments.filter(function (appointment) {
                if (searchInput.length > 0) {
                    return appointment.vin === searchInput;
                } else {
                    return appointment;
                }
            });
            setAppointments(newArray);
        } else {
            console.error(response);
        }
    }

    useEffect(() => {
        loadAppointments();
    }, [searchInput]);

    return (
        <div>
            <h1>Service History</h1>
            <div className="search">
                <form onSubmit={handleSubmit} id="submit-search-bar">
                    <input
                        onChange={handleSearchBarChange}
                        value={searchInput}
                        type="text"
                        placeholder="Search by VIN..."
                        name="search-bar"
                        id="header-search"
                        className="form-control"
                    />
                </form>
            </div>
            <table className="table table-striped">
                <thead>
                    <tr>
                        <th>VIN</th>
                        <th>Is VIP?</th>
                        <th>Customer</th>
                        <th>Date</th>
                        <th>Time</th>
                        <th>Technician</th>
                        <th>Reason</th>
                        <th>Status</th>
                    </tr>
                </thead>
                <tbody>
                    {appointments.map((appt) => {
                        const date = appt.date_time.split("T")[0];
                        const rawTime = appt.date_time.split("T")[1];
                        const time = rawTime.split("+")[0];
                        return (
                            <tr key={appt.id}>
                                <td>{appt.vin}</td>
                                <td>{appt.vip_status ? "Yes" : "No"}</td>
                                <td>{appt.customer}</td>
                                <td>{date}</td>
                                <td>{time}</td>
                                <td>
                                    {appt.technician.first_name + " " + appt.technician.last_name}
                                </td>
                                <td>{appt.reason}</td>
                                <td>{appt.status}</td>
                            </tr>
                        );
                    })}
                </tbody>
            </table>
        </div>
    );
}

export default AppointmentsHistory;
