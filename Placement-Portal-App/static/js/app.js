// JWT interceptor
axios.interceptors.request.use(config => {
    const token = localStorage.getItem('token');
    if (token) config.headers.Authorization = `Bearer ${token}`;
    return config;
});

const { createApp } = Vue;
const { createRouter, createWebHashHistory } = VueRouter;

const routes = [
    { path: '/', component: HomeView },
    { path: '/login', component: LoginView },
    { path: '/register', component: RegisterView },
    { path: '/admin', component: AdminDashboardView, meta: { requiresAuth: true, role: 'admin' } },
    { path: '/company', component: CompanyDashboardView, meta: { requiresAuth: true, role: 'company' } },
    { path: '/student', component: StudentDashboardView, meta: { requiresAuth: true, role: 'student' } }
];

const router = createRouter({
    history: createWebHashHistory(),
    routes,
});

router.beforeEach((to, from, next) => {
    const token = localStorage.getItem('token');
    const userStr = localStorage.getItem('user');

    if (to.meta.requiresAuth) {
        if (!token) return next('/login');
        const user = JSON.parse(userStr);
        if (to.meta.role && user.role !== to.meta.role) {
            return next('/');
        }
    } else if (to.path === '/login' || to.path === '/register') {
        if (token && userStr) {
            const user = JSON.parse(userStr);
            if (user.role === 'admin') return next('/admin');
            if (user.role === 'company') return next('/company');
            if (user.role === 'student') return next('/student');
        }
    }
    next();
});

const app = createApp({
    components: { NavbarComponent },
    template: `
        <div>
            <NavbarComponent />
            <div class="container mt-5 pt-4">
                <router-view></router-view>
            </div>
        </div>
    `
});

app.use(router);
app.mount('#app');
