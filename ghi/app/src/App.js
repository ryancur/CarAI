import { BrowserRouter, Routes, Route } from 'react-router-dom';
import MainPage from './MainPage';
import Navigation from './Nav';
import SalespersonForm from './SalespersonForm';
import SalespeopleList from './SalespeopleList';
import CustomerForm from './CustomerForm';
import CustomerList from './CustomerList';
import SaleForm from './SaleForm';
import SalesList from './SalesList';
import SalespersonHistory from './SalespersonHistory';
import TechnicianList from './TechnicianList';
import TechnicianForm from './TechnicianForm';
import AppointmentsList from './AppointmentsList';
import AppointmentForm from './AppointmentForm';
import AppointmentsHistory from './AppointmentsHistory';
import ManufacturersList from './ManufacturersList';
import ManufacturerForm from './ManufacturerForm';
import VehicleModelList from './VehicleModelList';
import VehicleModelForm from './VehicleModelForm';
import AutomobilesList from './AutomobilesList';
import AutomobileForm from './AutomobileForm';
import PredictAutoPriceForm from './PredictAutoPriceForm';
import Dashboard from './Dashboard';

function App() {
    return (
        <BrowserRouter>
            <Navigation />
            <div className="container">
                <Routes>
                    <Route path="/" element={<MainPage />} />
                    <Route path="manufacturers">
                        <Route path="/manufacturers" element={<ManufacturersList></ManufacturersList>} />
                        <Route path="create" element={<ManufacturerForm />} />
                    </Route>
                    <Route path="models">
                        <Route path="/models" element={<VehicleModelList></VehicleModelList>} />
                        <Route path="create" element={<VehicleModelForm />} />
                    </Route>
                    <Route path="automobiles">
                        <Route path="/automobiles" element={<AutomobilesList></AutomobilesList>} />
                        <Route path="create" element={<AutomobileForm />} />
                        <Route path="price" element={<PredictAutoPriceForm />} />
                    </Route>
                    <Route path="salespeople">
                        <Route path="/salespeople" element={<SalespeopleList></SalespeopleList>} />
                        <Route path="create" element={<SalespersonForm />} />
                    </Route>
                    <Route path="customers">
                        <Route path="/customers" element={<CustomerList></CustomerList>} />
                        <Route path="create" element={<CustomerForm />} />
                    </Route>
                    <Route path="sales">
                        <Route path="/sales" element={<SalesList></SalesList>} />
                        <Route path="create" element={<SaleForm />} />
                        <Route path="history" element={<SalespersonHistory />} />
                    </Route>
                    <Route path="technicians">
                        <Route path="/technicians" element={<TechnicianList></TechnicianList>} />
                        <Route path="create" element={<TechnicianForm />} />
                    </Route>
                    <Route path="appointments">
                        <Route path="/appointments" element={<AppointmentsList></AppointmentsList>} />
                        <Route path="create" element={<AppointmentForm />} />
                        <Route path="history" element={<AppointmentsHistory />} />
                    </Route>
                    <Route path="dashboard">
                        <Route path="/dashboard" element={<Dashboard></Dashboard>} />
                    </Route>
                </Routes>
            </div>
        </BrowserRouter>
    );
}

export default App;
