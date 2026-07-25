const AdminDashboardView = {
    template: `
    <div>
        <h2 class="mb-4 brand-title">Admin Dashboard</h2>
        
        <div class="row mb-5">
            <div class="col-md-4">
                <div class="glass-panel text-center py-4">
                    <h1 class="display-4 text-gradient fw-bold">{{ stats.students }}</h1>
                    <p class="mb-0 text-muted">Total Students</p>
                </div>
            </div>
            <div class="col-md-4">
                <div class="glass-panel text-center py-4">
                    <h1 class="display-4 text-gradient fw-bold">{{ stats.companies }}</h1>
                    <p class="mb-0 text-muted">Total Companies</p>
                </div>
            </div>
            <div class="col-md-4">
                <div class="glass-panel text-center py-4">
                    <h1 class="display-4 text-gradient fw-bold">{{ stats.drives }}</h1>
                    <p class="mb-0 text-muted">Placement Drives</p>
                </div>
            </div>
            <div class="col-md-6 mt-4">
                <div class="glass-panel text-center py-3">
                    <h2 class="text-gradient fw-bold mb-1">{{ stats.applications }}</h2>
                    <p class="mb-0 text-muted">Applications received</p>
                </div>
            </div>
            <div class="col-md-6 mt-4">
                <div class="glass-panel text-center py-3">
                    <h2 class="text-gradient fw-bold mb-1">{{ stats.selected }}</h2>
                    <p class="mb-0 text-muted">Students selected</p>
                </div>
            </div>
        </div>

        <ul class="nav nav-tabs mb-4 border-0" id="adminTabs">
            <li class="nav-item">
                <a class="nav-link text-white active bg-transparent border-bottom" data-bs-toggle="tab" href="#companies">Companies</a>
            </li>
            <li class="nav-item">
                <a class="nav-link text-white bg-transparent" data-bs-toggle="tab" href="#students">Students</a>
            </li>
            <li class="nav-item">
                <a class="nav-link text-white bg-transparent" data-bs-toggle="tab" href="#drives">Drives</a>
            </li>
            <li class="nav-item">
                <a class="nav-link text-white bg-transparent" data-bs-toggle="tab" href="#applications">All Applications</a>
            </li>
        </ul>

        <div class="tab-content" id="myTabContent">
            <!-- Companies Tab -->
            <div class="tab-pane fade show active" id="companies">
                <div class="glass-panel">
                    <div class="d-flex justify-content-between align-items-center mb-3">
                        <h4 class="mb-0">Companies</h4>
                        <input type="text" class="form-control w-25" placeholder="Search companies..." v-model="searchCompany">
                    </div>
                    <table class="table table-dark table-hover table-borderless">
                        <thead>
                            <tr>
                                <th>Name</th>
                                <th>Email</th>
                                <th>Status</th>
                                <th>Actions</th>
                            </tr>
                        </thead>
                        <tbody>
                            <tr v-for="company in filteredCompanies" :key="company.id">
                                <td>{{ company.name }}</td>
                                <td>{{ company.email }}</td>
                                <td>
                                    <span class="badge" :class="company.is_approved ? 'bg-success' : 'bg-warning'">
                                        {{ company.is_approved ? 'Approved' : 'Pending' }}
                                    </span>
                                </td>
                                <td>
                                    <button class="btn btn-sm btn-success me-2" v-if="!company.is_approved" @click="approveCompany(company.id)">Approve</button>
                                    <button class="btn btn-sm btn-danger me-2" v-if="!company.is_approved" @click="rejectCompany(company.id)">Reject</button>
                                    <button class="btn btn-sm" :class="company.is_active ? 'btn-danger' : 'btn-info'" @click="toggleActive(company.id)">
                                        {{ company.is_active ? 'Deactivate' : 'Activate' }}
                                    </button>
                                </td>
                            </tr>
                        </tbody>
                    </table>
                </div>
            </div>

            <!-- Students Tab -->
            <div class="tab-pane fade" id="students">
                <div class="glass-panel">
                    <div class="d-flex justify-content-between align-items-center mb-3">
                        <h4 class="mb-0">Students</h4>
                        <input type="text" class="form-control w-25" placeholder="Search students..." v-model="searchStudent">
                    </div>
                    <table class="table table-dark table-hover table-borderless">
                        <thead>
                            <tr>
                                <th>Name</th>
                                <th>Email</th>
                                <th>Actions</th>
                            </tr>
                        </thead>
                        <tbody>
                            <tr v-for="student in filteredStudents" :key="student.id">
                                <td>{{ student.name }}</td>
                                <td>{{ student.email }}</td>
                                <td>
                                    <button class="btn btn-sm" :class="student.is_active ? 'btn-danger' : 'btn-info'" @click="toggleActive(student.id)">
                                        {{ student.is_active ? 'Deactivate' : 'Activate' }}
                                    </button>
                                </td>
                            </tr>
                        </tbody>
                    </table>
                </div>
            </div>

            <!-- Drives Tab -->
            <div class="tab-pane fade" id="drives">
                <div class="glass-panel">
                    <div class="d-flex justify-content-between align-items-center mb-3">
                        <h4 class="mb-0">Placement Drives</h4>
                        <input type="text" class="form-control w-25" placeholder="Search drives..." v-model="searchDrive">
                    </div>
                    <table class="table table-dark table-hover table-borderless">
                        <thead>
                            <tr>
                                <th>Company</th>
                                <th>Job Title</th>
                                <th>Deadline</th>
                                <th>Applicants</th>
                                <th>Status</th>
                                <th>Action</th>
                            </tr>
                        </thead>
                        <tbody>
                            <tr v-for="drive in filteredDrives" :key="drive.id">
                                <td>{{ drive.company_name }}</td>
                                <td>{{ drive.job_title }}</td>
                                <td>{{ new Date(drive.application_deadline).toLocaleDateString() }}</td>
                                <td>{{ drive.applicant_count }}</td>
                                <td>
                                    <span class="badge" :class="{'bg-success': drive.status === 'Approved', 'bg-warning': drive.status === 'Pending', 'bg-danger': drive.status === 'Rejected'}">
                                        {{ drive.status }}
                                    </span>
                                </td>
                                <td>
                                    <button class="btn btn-sm btn-success me-2" v-if="drive.status === 'Pending'" @click="approveDrive(drive.id)">Approve</button>
                                    <button class="btn btn-sm btn-danger" v-if="drive.status === 'Pending'" @click="rejectDrive(drive.id)">Reject</button>
                                </td>
                            </tr>
                        </tbody>
                    </table>
                </div>
            </div>

            <!-- Applications Tab -->
            <div class="tab-pane fade" id="applications">
                <div class="glass-panel">
                    <h4>All Student Applications</h4>
                    <table class="table table-dark table-hover table-borderless">
                        <thead>
                            <tr>
                                <th>Student</th>
                                <th>Company</th>
                                <th>Job Title</th>
                                <th>Date</th>
                                <th>Status</th>
                            </tr>
                        </thead>
                        <tbody>
                            <tr v-for="app in applications" :key="app.id">
                                <td>{{ app.student_name }}</td>
                                <td>{{ app.company_name }}</td>
                                <td>{{ app.job_title }}</td>
                                <td>{{ new Date(app.application_date).toLocaleDateString() }}</td>
                                <td><span class="badge bg-secondary">{{ app.status }}</span></td>
                            </tr>
                        </tbody>
                    </table>
                </div>
            </div>
        </div>
    </div>
    `,
    data() {
        return {
            stats: { students: 0, companies: 0, drives: 0, applications: 0, selected: 0 },
            companies: [],
            students: [],
            drives: [],
            applications: [],
            searchCompany: '',
            searchStudent: '',
            searchDrive: ''
        }
    },
    computed: {
        filteredCompanies() {
            return this.companies.filter(c => c.name.toLowerCase().includes(this.searchCompany.toLowerCase()) || c.email.toLowerCase().includes(this.searchCompany.toLowerCase()));
        },
        filteredStudents() {
            return this.students.filter(s => s.name.toLowerCase().includes(this.searchStudent.toLowerCase()) || s.email.toLowerCase().includes(this.searchStudent.toLowerCase()));
        },
        filteredDrives() {
            const query = this.searchDrive.toLowerCase();
            return this.drives.filter(d => d.job_title.toLowerCase().includes(query) || d.company_name.toLowerCase().includes(query));
        }
    },
    created() {
        this.fetchData();
    },
    methods: {
        async fetchData() {
            try {
                const [statsRes, compRes, studRes, driveRes, appRes] = await Promise.all([
                    axios.get('/api/admin/dashboard'),
                    axios.get('/api/admin/companies'),
                    axios.get('/api/admin/students'),
                    axios.get('/api/admin/drives'),
                    axios.get('/api/admin/applications')
                ]);
                this.stats = statsRes.data;
                this.companies = compRes.data;
                this.students = studRes.data;
                this.drives = driveRes.data;
                this.applications = appRes.data;
            } catch (err) {
                console.error(err);
            }
        },
        async approveCompany(id) {
            await axios.post('/api/admin/approve_company/' + id);
            this.fetchData();
        },
        async rejectCompany(id) {
            if (confirm("Are you sure you want to reject and remove this company?")) {
                await axios.post('/api/admin/reject_company/' + id);
                this.fetchData();
            }
        },
        async toggleActive(id) {
            await axios.post('/api/admin/toggle_active/' + id);
            this.fetchData();
        },
        async approveDrive(id) {
            await axios.post('/api/admin/approve_drive/' + id);
            this.fetchData();
        },
        async rejectDrive(id) {
            await axios.post('/api/admin/reject_drive/' + id);
            this.fetchData();
        }
    }
}
