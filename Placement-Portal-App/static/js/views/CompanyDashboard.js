const CompanyDashboardView = {
    template: `
    <div>
        <h2 class="mb-4 brand-title">Company Dashboard</h2>
        
        <div v-if="!user || !user.is_approved" class="alert alert-warning">
            Your profile is currently waiting for Admin approval. You cannot create placement drives yet.
        </div>

        <div class="row">
            <div class="col-md-4 mb-4">
                <div class="glass-panel mb-4">
                    <div class="d-flex align-items-center justify-content-between mb-3">
                        <h4 class="mb-0">Company Profile</h4>
                        <span class="badge" :class="user && user.is_approved ? 'bg-success' : 'bg-warning'">
                            {{ user && user.is_approved ? 'Approved' : 'Pending' }}
                        </span>
                    </div>
                    <form @submit.prevent="updateProfile">
                        <div class="mb-2">
                            <label class="form-label">HR Contact</label>
                            <input type="text" class="form-control" v-model="profile.hr_contact" placeholder="Name or email">
                        </div>
                        <div class="mb-3">
                            <label class="form-label">Website</label>
                            <input type="url" class="form-control" v-model="profile.website" placeholder="https://example.com">
                        </div>
                        <button type="submit" class="btn btn-outline-primary w-100">Save Profile</button>
                    </form>
                </div>
                <div class="glass-panel">
                    <h4>Create Drive</h4>
                    <form @submit.prevent="createDrive">
                        <div class="mb-2">
                            <label class="form-label">Job Title</label>
                            <input type="text" class="form-control" v-model="form.job_title" required>
                        </div>
                        <div class="mb-2">
                            <label class="form-label">Job Description</label>
                            <textarea class="form-control" v-model="form.job_description" required></textarea>
                        </div>
                        <div class="mb-2">
                            <label class="form-label">Eligibility Branch</label>
                            <input type="text" class="form-control" v-model="form.eligibility_branch" placeholder="Data Science, or leave blank for Any">
                            <small class="text-muted">Use a branch name, not a graduation year.</small>
                        </div>
                        <div class="mb-2">
                            <label class="form-label">Min CGPA</label>
                            <input type="number" min="0" max="10" step="0.1" class="form-control" v-model="form.eligibility_cgpa">
                        </div>
                        <div class="mb-2">
                            <label class="form-label">Minimum Year</label>
                            <input type="number" min="1" max="8" class="form-control" v-model="form.eligibility_year" required>
                        </div>
                        <div class="mb-2">
                            <label class="form-label">Deadline</label>
                            <input type="date" class="form-control" v-model="form.application_deadline" required>
                            <small class="text-muted">Applications remain open until 11:59 PM on this date.</small>
                        </div>
                        <button type="submit" class="btn btn-primary-gradient mt-2 w-100" :disabled="!user.is_approved">
                            Create Drive
                        </button>
                    </form>
                </div>
            </div>
            
            <div class="col-md-8">
                <div class="glass-panel mb-4">
                    <h4>Your Placement Drives</h4>
                    <table class="table table-dark table-hover table-borderless">
                        <thead>
                            <tr>
                                <th>Job Title</th>
                                <th>Deadline</th>
                                <th>Applicants</th>
                                <th>Status</th>
                                <th>Actions</th>
                            </tr>
                        </thead>
                        <tbody>
                            <tr v-for="drive in drives" :key="drive.id">
                                <td>{{ drive.job_title }}</td>
                                <td>{{ new Date(drive.application_deadline).toLocaleDateString() }}</td>
                                <td>{{ drive.applicant_count }}</td>
                                <td>
                                    <span class="badge" :class="{'bg-success': drive.status === 'Approved', 'bg-warning': drive.status === 'Pending', 'bg-danger': drive.status === 'Rejected'}">
                                        {{ drive.status }}
                                    </span>
                                </td>
                                <td>
                                    <button class="btn btn-sm btn-info" @click="viewApplications(drive.id)">View Apps</button>
                                    <button v-if="drive.status === 'Approved'" class="btn btn-sm btn-outline-warning ms-2" @click="closeDrive(drive.id)">Close</button>
                                </td>
                            </tr>
                        </tbody>
                    </table>
                </div>

                <div class="glass-panel" v-if="selectedDriveApps">
                    <h4>Applicants</h4>
                    <table class="table table-dark table-hover table-borderless">
                        <thead>
                            <tr>
                                <th>Name</th>
                                <th>Branch</th>
                                <th>CGPA</th>
                                <th>Resume</th>
                                <th>Status</th>
                                <th>Action</th>
                            </tr>
                        </thead>
                        <tbody>
                            <tr v-for="app in selectedDriveApps" :key="app.id">
                                <td>{{ app.student_name }}</td>
                                <td>{{ app.student_profile ? app.student_profile.branch : '-' }}</td>
                                <td>{{ app.student_profile ? app.student_profile.cgpa : '-' }}</td>
                                <td>
                                    <a v-if="app.student_profile && app.student_profile.resume_link" 
                                       :href="'/' + app.student_profile.resume_link" target="_blank" class="text-info text-decoration-none">
                                        <i class="bi bi-file-earmark-pdf"></i> View
                                    </a>
                                    <span v-else class="text-muted small">N/A</span>
                                </td>
                                <td>{{ app.status }}</td>
                                <td>
                                    <select class="form-select form-select-sm" :value="app.status" @change="updateStatus(app.id, $event.target.value)">
                                        <option value="Applied">Applied</option>
                                        <option value="Shortlisted">Shortlisted</option>
                                        <option value="Interview Scheduled">Interview Scheduled</option>
                                        <option value="Selected">Selected</option>
                                        <option value="Rejected">Rejected</option>
                                    </select>
                                </td>
                            </tr>
                            <tr v-if="selectedDriveApps.length === 0">
                                <td colspan="6" class="text-center text-muted">No applications yet.</td>
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
            user: null,
            profile: { hr_contact: '', website: '' },
            drives: [],
            selectedDriveApps: null,
            form: {
                job_title: '',
                job_description: '',
                eligibility_branch: '',
                eligibility_cgpa: 0,
                eligibility_year: 1,
                application_deadline: ''
            }
        }
    },
    created() {
        this.user = JSON.parse(localStorage.getItem('user'));
        this.fetchProfile();
        this.fetchDrives();
    },
    methods: {
        async fetchDrives() {
            try {
                const res = await axios.get('/api/company/drives');
                this.drives = res.data;
            } catch (err) { console.error(err); }
        },
        async fetchProfile() {
            try {
                const res = await axios.get('/api/company/profile');
                if (res.data) this.profile = res.data;
            } catch (err) { console.error(err); }
        },
        async updateProfile() {
            try {
                await axios.post('/api/company/profile', this.profile);
                alert('Company profile updated');
            } catch (err) { alert(err.response?.data?.error || 'Could not update profile'); }
        },
        async createDrive() {
            try {
                // A date-only field is stored as the end of the selected day.
                const dt = new Date(this.form.application_deadline + 'T23:59:59');
                const payload = { ...this.form, application_deadline: dt.toISOString() };
                await axios.post('/api/company/drives', payload);
                this.fetchDrives();
                alert('Drive Created');
            } catch (err) { alert(err.response?.data?.error || 'Error'); }
        },
        async viewApplications(drive_id) {
            try {
                const res = await axios.get('/api/company/drives/' + drive_id + '/applications');
                this.selectedDriveApps = res.data;
            } catch (err) { console.error(err); }
        },
        async updateStatus(app_id, status) {
            try {
                await axios.post('/api/company/applications/' + app_id + '/status', { status });
                alert('Status Updated');
            } catch (err) { console.error(err); }
        },
        async closeDrive(drive_id) {
            if (!confirm('Close this drive? Students will no longer be able to apply.')) return;
            try {
                await axios.post('/api/company/drives/' + drive_id + '/close');
                this.fetchDrives();
            } catch (err) { alert(err.response?.data?.error || 'Could not close drive'); }
        }
    }
}
