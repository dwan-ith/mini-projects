const StudentDashboardView = {
    template: `
    <div>
        <h2 class="mb-4 brand-title">Student Dashboard</h2>
        
        <div class="row">
            <div class="col-md-4 mb-4">
                <div class="glass-panel">
                    <h4>My Profile</h4>
                    <form @submit.prevent="updateProfile">
                        <div class="mb-2">
                            <label class="form-label">CGPA</label>
                            <input type="number" step="0.1" class="form-control" v-model="profile.cgpa">
                        </div>
                        <div class="mb-2">
                            <label class="form-label">Branch</label>
                            <input type="text" class="form-control" v-model="profile.branch">
                        </div>
                        <div class="mb-2">
                            <label class="form-label">Year</label>
                            <input type="number" class="form-control" v-model="profile.year">
                        </div>
                        <div class="mb-2">
                            <label class="form-label">Resume Upload</label>
                            <input type="file" ref="resumeFile" class="form-control mb-2" accept=".pdf,.doc,.docx">
                            <button type="button" class="btn btn-sm btn-outline-info w-100" @click="uploadResume">Upload Resume</button>
                            <small v-if="profile.resume_link" class="text-success d-block mt-1">
                                <i class="bi bi-check-circle"></i> Resume on file
                            </small>
                        </div>
                        <button type="submit" class="btn btn-primary-gradient mt-3 w-100">Update Profile</button>
                    </form>
                </div>
                
                <div class="glass-panel mt-4">
                    <h4>Export Data</h4>
                    <p class="text-muted small">Receive a CSV of all your applications via background job.</p>
                    <button class="btn btn-outline-info w-100" :disabled="exporting" @click="exportCSV">{{ exporting ? 'Preparing export...' : 'Export Applications (CSV)' }}</button>
                    <small v-if="exportMessage" class="d-block mt-2" :class="exportError ? 'text-danger' : 'text-success'">{{ exportMessage }}</small>
                </div>
            </div>
            
            <div class="col-md-8">
                <ul class="nav nav-tabs border-0 mb-4">
                    <li class="nav-item">
                        <a class="nav-link bg-transparent text-white active border-bottom" data-bs-toggle="tab" href="#available-drives">Available Drives</a>
                    </li>
                    <li class="nav-item">
                        <a class="nav-link bg-transparent text-white" data-bs-toggle="tab" href="#my-applications">My Applications</a>
                    </li>
                </ul>

                <div class="tab-content">
                    <div class="tab-pane fade show active" id="available-drives">
                        <div class="mb-3">
                            <input type="text" class="form-control w-50" placeholder="Search by Job Job Title or Company..." v-model="searchDrive">
                        </div>
                        <div class="row">
                            <div class="col-md-6 mb-3" v-for="drive in filteredDrives" :key="drive.id">
                                <div class="glass-panel h-100 d-flex flex-column">
                                    <h5 class="text-gradient">{{ drive.job_title }}</h5>
                                    <p class="mb-1 text-white fw-bold">{{ drive.company_name }}</p>
                                    <p class="mb-2 small text-muted">{{ drive.job_description.substring(0, 50) }}...</p>
                                    <ul class="list-unstyled small mb-3 flex-grow-1">
                                        <li><i class="bi bi-mortarboard me-1"></i>Min CGPA: {{ drive.eligibility_cgpa }}</li>
                                        <li><i class="bi bi-book me-1"></i>Branch: {{ drive.eligibility_branch }}</li>
                                        <li><i class="bi bi-calendar me-1"></i>Deadline: {{ new Date(drive.application_deadline).toLocaleDateString() }}</li>
                                    </ul>
                                    <small v-if="!drive.is_eligible" class="text-warning d-block mb-2">{{ drive.eligibility_message }}</small>
                                    <button class="btn btn-sm w-100 mt-auto" :class="drive.has_applied ? 'btn-success' : 'btn-primary-gradient'" :disabled="drive.has_applied || !drive.is_eligible || applyingDriveId === drive.id" @click="apply(drive.id)">
                                        {{ drive.has_applied ? 'Applied' : (applyingDriveId === drive.id ? 'Applying...' : (drive.is_eligible ? 'Apply Now' : 'Not Eligible')) }}
                                    </button>
                                </div>
                            </div>
                            <div class="col-12" v-if="filteredDrives.length === 0">
                                <p class="text-muted">No drives match your search criteria.</p>
                            </div>
                        </div>
                    </div>
                    
                    <div class="tab-pane fade" id="my-applications">
                        <div class="glass-panel">
                            <table class="table table-dark table-hover table-borderless">
                                <thead>
                                    <tr>
                                        <th>Company</th>
                                        <th>Job Title</th>
                                        <th>Applied On</th>
                                        <th>Status</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    <tr v-for="app in applications" :key="app.id">
                                        <td>{{ app.company_name }}</td>
                                        <td>{{ app.job_title }}</td>
                                        <td>{{ new Date(app.application_date).toLocaleDateString() }}</td>
                                        <td>
                                            <span class="badge" 
                                                  :class="{'bg-info': app.status === 'Applied', 'bg-primary': app.status === 'Interview Scheduled', 'bg-warning': app.status === 'Shortlisted', 'bg-success': app.status === 'Selected', 'bg-danger': app.status === 'Rejected'}">
                                                {{ app.status }}
                                            </span>
                                        </td>
                                    </tr>
                                    <tr v-if="applications.length === 0">
                                        <td colspan="4" class="text-center text-muted">No applications yet.</td>
                                    </tr>
                                </tbody>
                            </table>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>
    `,
    data() {
        return {
            profile: { cgpa: 0, branch: '', year: 1, resume_link: '' },
            drives: [],
            applications: [],
            searchDrive: '',
            exporting: false,
            exportMessage: '',
            exportError: false,
            applyingDriveId: null
        }
    },
    computed: {
        filteredDrives() {
            return this.drives.filter(d =>
                (d.job_title && d.job_title.toLowerCase().includes(this.searchDrive.toLowerCase())) ||
                (d.company_name && d.company_name.toLowerCase().includes(this.searchDrive.toLowerCase()))
            );
        }
    },
    created() {
        this.fetchProfile();
        this.fetchDrives();
        this.fetchApplications();
    },
    methods: {
        async fetchProfile() {
            try {
                const res = await axios.get('/api/student/profile');
                if (res.data) this.profile = res.data;
            } catch (err) { console.error(err); }
        },
        async updateProfile() {
            try {
                await axios.post('/api/student/profile', this.profile);
                alert('Profile Updated');
            } catch (err) { console.error(err); }
        },
        async uploadResume() {
            const fileInput = this.$refs.resumeFile;
            if (fileInput.files.length === 0) {
                alert('Please select a file first.');
                return;
            }
            const file = fileInput.files[0];
            const formData = new FormData();
            formData.append('file', file);
            try {
                const res = await axios.post('/api/student/upload_resume', formData, {
                    headers: { 'Content-Type': 'multipart/form-data' }
                });
                alert('Resume uploaded!');
                this.profile.resume_link = res.data.filepath;
            } catch (err) {
                alert(err.response?.data?.error || 'Upload error');
            }
        },
        async fetchDrives() {
            try {
                const res = await axios.get('/api/student/drives');
                this.drives = res.data;
            } catch (err) { console.error(err); }
        },
        async fetchApplications() {
            try {
                const res = await axios.get('/api/student/applications');
                this.applications = res.data;
            } catch (err) { console.error(err); }
        },
        async apply(drive_id) {
            try {
                this.applyingDriveId = drive_id;
                await axios.post('/api/student/applications', { drive_id });
                alert('Applied Successfully');
                const drive = this.drives.find(item => item.id === drive_id);
                if (drive) drive.has_applied = true;
                this.fetchApplications();
            } catch (err) { alert(err.response?.data?.error || 'Could not apply to this drive.'); }
            finally { this.applyingDriveId = null; }
        },
        async exportCSV() {
            try {
                this.exporting = true; this.exportError = false; this.exportMessage = 'Export queued.';
                const res = await axios.post('/api/student/export');
                this.pollExport(res.data.job.id);
            } catch (err) {
                this.exporting = false; this.exportError = true;
                this.exportMessage = err.response?.data?.error || 'Could not start export.';
            }
        },
        async pollExport(jobId) {
            try {
                const res = await axios.get('/api/student/export/' + jobId);
                const job = res.data;
                if (job.status === 'Completed') {
                    this.exporting = false; this.exportMessage = 'Export is ready.';
                    window.open(job.download_url, '_blank');
                } else if (job.status === 'Failed') {
                    this.exporting = false; this.exportError = true;
                    this.exportMessage = job.error_message || 'Export failed.';
                } else {
                    setTimeout(() => this.pollExport(jobId), 1500);
                }
            } catch (err) {
                this.exporting = false; this.exportError = true; this.exportMessage = 'Could not check export status.';
            }
        }
    }
}
