const HomeView = {
    template: `
    <div class="text-center py-5">
        <div class="py-5">
            <i class="bi bi-rocket-takeoff" style="font-size: 4rem; color: #3b82f6;"></i>
            <h1 class="mt-4 fw-bold" style="font-size: 2.5rem;">Placement Portal</h1>
            <p class="text-muted mt-3 mb-5" style="max-width: 500px; margin: 0 auto;">
                A unified platform for students, companies, and administrators to manage campus placements efficiently.
            </p>
            <div class="d-flex justify-content-center gap-3">
                <router-link to="/login" class="btn btn-primary-gradient px-4 py-2">
                    <i class="bi bi-box-arrow-in-right me-2"></i>Login
                </router-link>
                <router-link to="/register" class="btn btn-outline-light px-4 py-2">
                    <i class="bi bi-person-plus me-2"></i>Register
                </router-link>
            </div>
        </div>

        <div class="row mt-5 g-4" style="max-width: 900px; margin: 0 auto;">
            <div class="col-md-4">
                <div class="glass-panel text-start h-100">
                    <i class="bi bi-person-badge fs-2 text-primary mb-3 d-block"></i>
                    <h5 class="fw-semibold">Students</h5>
                    <p class="text-muted small mb-0">Browse approved placement drives, apply with one click, and track your application status in real-time.</p>
                </div>
            </div>
            <div class="col-md-4">
                <div class="glass-panel text-start h-100">
                    <i class="bi bi-building fs-2 text-primary mb-3 d-block"></i>
                    <h5 class="fw-semibold">Companies</h5>
                    <p class="text-muted small mb-0">Post placement drives, review student applications, and manage the entire recruitment pipeline.</p>
                </div>
            </div>
            <div class="col-md-4">
                <div class="glass-panel text-start h-100">
                    <i class="bi bi-shield-check fs-2 text-primary mb-3 d-block"></i>
                    <h5 class="fw-semibold">Admin</h5>
                    <p class="text-muted small mb-0">Approve companies and drives, monitor all activity, and generate placement reports for the institute.</p>
                </div>
            </div>
        </div>
    </div>
    `
}
