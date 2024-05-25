//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
//

#define REAL_TIME_TRANSPORT_DEBUG

#include <gtest/gtest.h>

#include <RealTimeTransport/Models/ResonantLevel.h>
#include <RealTimeTransport/Utility.h>

#include <SciCore/Utility.h>

using namespace SciCore;
using namespace RealTimeTransport;

class FakeModel : public Model
{
  public:
    ~FakeModel() noexcept
    {
    }

    int dimHilbertSpace() const noexcept
    {
        return 2;
    }

    int numStates() const noexcept
    {
        return 12;
    }
    int numChannels() const noexcept
    {
        return 0;
    }

    int numReservoirs() const
    {
        return 0;
    }

    OperatorType H() const
    {
        return OperatorType::Zero(4, 4);
    }

    OperatorType P() const
    {
        return OperatorType::Zero(4, 4);
    }

    OperatorType d(int) const
    {
        return OperatorType::Zero(4, 4);
    }

    SciCore::Complex coupling(int, int, int) const
    {
        return 0.0;
    }

    const SciCore::RealVector& temperatures() const noexcept
    {
        static SciCore::RealVector T;
        return T;
    }

    const SciCore::RealVector& chemicalPotentials() const noexcept
    {
        static SciCore::RealVector mu;
        return mu;
    }

    const std::vector<int>& blockDimensions() const noexcept
    {
        return blocks;
    }

    std::unique_ptr<Model> copy() const override
    {
        throw Error("Not implemented");
    }

    bool isEqual(const Model&) const
    {
        throw Error("Not implemented");
    }

    std::vector<int> blocks;
};

TEST(singleToMultiIndexAndBack, Test)
{
    FakeModel model;
    int index = 0;
    for (int eta = 0; eta <= 1; ++eta)
    {
        for (int l = 0; l < model.numStates(); ++l)
        {
            int computedIndex = singleToMultiIndex(static_cast<Eta>(eta), l, &model);
            EXPECT_EQ(index, computedIndex);

            auto [eta2, l2] = multiToSingleIndices(index, &model);
            EXPECT_EQ(static_cast<Eta>(eta), eta2);
            EXPECT_EQ(l, l2);

            ++index;
        }
    }
}

TEST(defaultVectorizeOperatorize, Test)
{
    FakeModel model;

    Model::OperatorType M{
        {2.0, -4.0, Complex(1.0, 2.0)},
        {9.9, 2.0, 3.4},
        {Complex(1.2, 2.1), 9.3, 2.7}
    };

    Model::SupervectorType V{
        {2.0, 9.9, Complex(1.2, 2.1), -4.0, 2.0, 9.3, Complex(1.0, 2.0), 3.4, 2.7}
    };

    Model::SupervectorType v = model.vectorize(M);
    EXPECT_EQ(v, V);

    Model::OperatorType m = model.operatorize(V);
    EXPECT_EQ(m, M);
}

TEST(expm1, TestVsExactResults)
{
    Complex I(0, 1);
    Matrix A{
        {1. / 13.,      9. / 7.,          -3. + I,       0.,      11.},
        {      1.,  1. / 3. * I, 1. + 1. / 3. * I,  -2. * I,       2.},
        {1. / 10.,     1. / 20.,         1. / 30., 1. / 40., 1. / 50.},
        { 9. / 7., I * 1. / 20.,         1. / 30., 1. / 40., 1. / 50.},
        {      1.,          -5.,               3.,        I,       2.}
    };

    struct TestData
    {
        Real t;
        Real errorGoal;
        Matrix result;
    };

    std::vector testData{
        TestData{0, 0.0, Matrix::Zero(5, 5)},
        TestData{
                 0.1, 5e-14,
                 Matrix{
                {Complex(0.0631794881069891769814733677001, 0.0031060647073898831891828389022),
                 Complex(-0.166569966923285103856467163671, -0.000844062135897423390957904534),
                 Complex(-0.131949936167045155160904853434, 0.101265344031202353174464489944),
                 Complex(-0.000258566633705122227557207733, 0.0661487833303238449091348568461),
                 Complex(1.237337066093883993591209986063, 0.001048164119393590119057342041)},
                {Complex(0.1120186298071703465744678929497, -0.0104311790486408754349084220891),
                 Complex(-0.0561394964909413182396817497357, 0.0323517390614275114903202403787),
                 Complex(0.1200295370248265976694358369535, 0.040609131878720552168322772132),
                 Complex(0.00341672765421133526402541216, -0.184011302496746405726293304464),
                 Complex(0.281216496495995943651781121612, -0.000738206501854128734071458389)},
                {Complex(0.01076806447736072639995361420468, -4.05848466159847677779211673e-6),
                 Complex(0.00406376888908846451483078665922, 0.00008257378012824757694438678906),
                 Complex(0.00296812071973984221024864468865, 0.00060412906952386013946396163962),
                 Complex(0.00251247852338336025222752301112, -0.00013280793757010801505087032823),
                 Complex(0.00887227336622325764170730082519, 4.58856016386919274374660111e-6)},
                {Complex(0.1317824402998459943089107386625, 0.0003758850063667766178123821515),
                 Complex(-0.00476808187699866777840484728501, 0.00491242050448163827990095337426),
                 Complex(-0.00853806393049664276974307433451, 0.00678555235486321796614517847808),
                 Complex(0.0029727023191140071844737510834, 0.00271990861661621400436737554874),
                 Complex(0.0787876772229927256437165267461, 0.0006649903967744674798819656231)},
                {Complex(0.0859559859161211099399210624244, 0.0089100316441243979818597634031),
                 Complex(-0.547073803138036302878660344717, -0.008824227730844101049133187905),
                 Complex(0.292182354739997728407901541449, -0.004994385775932424255585002014),
                 Complex(-0.000273629012998868621235660466, 0.163931731199351989714153947517),
                 Complex(0.218521996807505441214445501226, 0.002721198658712438434240667195)}}},
        TestData{
                 0.01, 5e-14,
                 Matrix{
                {Complex(0.001364550458645345864848919210301, 6.971348346938793868756541138e-6),
                 Complex(0.01008604688457374234385320172474, 0.00002086442621872823577946463753),
                 Complex(-0.0283045262260243809779503866636, 0.0100259278185918698473717454131),
                 Complex(-3.469106023925647080663229012e-6, 0.000445416768648702681162978898731),
                 Complex(0.111279929525918501304128779608, 1.4211858714224426058153684e-6)},
                {Complex(0.01011011479074894325387688113499, -0.0001097122306197380190647315054),
                 Complex(-0.00044588417757748636393369129604, 0.00333274043777708655996241571949),
                 Complex(0.01015147981664554086616011252847, 0.00339850033737519102464896106097),
                 Complex(0.000034501110382101143375855745, -0.0198965984889434148709347408635),
                 Complex(0.0207571232778899861475375772092, 0.0000278405383273212867337994624)},
                {Complex(0.001005868194002490396211098263545, -1.1539023247248679750519284e-8),
                 Complex(0.000500492750535824982849823724241, 8.97542131437532908107068103e-7),
                 Complex(0.000324498272187556149219723085158, 5.85689638694028758600619021e-6),
                 Complex(0.000250078142008881152547056046816, -3.79637840644647481707181395e-6),
                 Complex(0.000262692870290598772657802247813, 5.823539124670924467305375e-9)},
                {Complex(0.01286742864958390833344136606713, 2.55063429714360645523161145e-6),
                 Complex(0.000065023745624844395536429416513, 0.000500075942193367118257803563959),
                 Complex(0.000149975325666477404422168017865, 0.000066926995180070356541359997902),
                 Complex(0.000255041406366479145458176394196, 2.930287525942568664538735066e-6),
                 Complex(0.000914751947000237171457437190932, 5.13213125205726084439871614e-6)},
                {Complex(0.00986817613954756345873201960114, 0.000066610260695528635690313334),
                 Complex(-0.0504354428981395462907155894025, -0.0000835782977099356251244850703),
                 Complex(0.0299072162204190003847515607339, -0.0000334389444005580914296137197),
                 Complex(3.17764679120756269167833481e-6, 0.01060502092547473063258864520629),
                 Complex(0.020246927989656431127060779989, 2.9184771116992980370667154e-6)}}},
        TestData{
                 0.00044, 5e-15, // expm1 used specialized algorithm for very small t, therefore we get better error estimates
            Matrix{
                {Complex(0.0000350065820756986685944719195043, 9.8425096109622094296475996e-9),
                 Complex(0.000560383773395335622474840720732, 4.6065870313284977518767586e-8),
                 Complex(-0.001316713360985019698642696328825, 0.000440052075738478280975915967368),
                 Complex(-7.234731160957992408798074194e-9, 8.20232537458261218586543192904e-7),
                 Complex(0.00484245565264772570395128177686, 1.96847780645553527563557e-9)},
                {Complex(0.000440210835331362505453578607711, -2.13376553644963033898868977e-7),
                 Complex(-8.40804556790586387384376347e-7, 0.000146668155286865464549353902636),
                 Complex(0.000440283327913483781482074705606, 0.000146790467895345157557400542851),
                 Complex(6.6946107163556235520408458e-8, -0.000879809972321136786422889781308),
                 Complex(0.000881454457172595321165412279404, 6.0981678987286220519910848e-8)},
                {Complex(0.0000440109726415745246268084064448, -1.0378222091720454157161e-12),
                 Complex(0.0000220028408187330645673471144471, 1.7345567065435611615515846e-9),
                 Complex(0.0000146485111914062522790941214768, 1.12953542637839811252117142e-8),
                 Complex(0.00001100014160930120381384589436765, -7.72719939792456194122364958e-9),
                 Complex(8.92019866972045304794311905647e-6, 5.1809751639053391275068e-13)},
                {Complex(0.000565729449205488018108789481085, 4.843777533783950464012081e-9),
                 Complex(1.478773171400987873545000343e-7, 0.0000220001230705822664008752563797),
                 Complex(0.0000142982950157798344187619762301, 1.293093928410265712376441378e-7),
                 Complex(0.00001100981846640784196182678659645, 2.09445742372448218930106084e-9),
                 Complex(0.00001017348313283650463206065557668, 9.69107627815625878307580371e-9)},
                {Complex(0.000439746031392954725686723922419, 1.24653831018362773304651086e-7),
                 Complex(-0.00220083427165457907834800068247, -1.6135252185972875719334773e-7),
                 Complex(0.001319816273491710143628393107279, -6.1456889905469682514542945e-8),
                 Complex(7.211722713613936850869437e-9, 0.000441164340515732731901214976922),
                 Complex(0.000880489296643017887159999643446, 2.093537712760761705123397e-9)}}},
        TestData{
                 1e-4, 5e-15, // expm1 used specialized algorithm for very small t, therefore we get better error estimates
            Matrix{
                {Complex(7.75226160403991309358701235731e-6, 5.0190549874537567631419037e-10),
                 Complex(0.0001282961546994967815740515040949, 2.3898072353930597309408127e-9),
                 Complex(-0.000299830227621215041299714107338, 0.000100002693168692770376729952701),
                 Complex(-3.747028063564220089116358407e-10, 4.22904724785708703255628758793e-8),
                 Complex(0.001100126800110362431197449247648, 1.00379899768865358940353e-10)},
                {Complex(0.0001000108858975107377857403269574, -1.10232884713345432259875441e-8),
                 Complex(-4.33890279150355568485961876e-8, 0.0000333334152012802223429258456015),
                 Complex(0.0001000146164734268341098693676119, 0.000033339723542455517041210640345),
                 Complex(3.45824841659638873234935e-9, -0.000199990202923957923981818535253),
                 Complex(0.000200075106117998610041932220062, 3.162851939521514556457343e-9)},
                {Complex(0.00001000056605301321138496102542633, -1.22057152686221541902e-14),
                 Complex(5.00015017845356547455665146079e-6, 8.958598805137331793286195e-11),
                 Complex(3.33239362256927260288980005716e-6, 5.833570629111731950366566e-10),
                 Complex(2.50000729686173861903067836437e-6, -3.9980299897066532484693211e-10),
                 Complex(2.00620646402762394826768637712e-6, 6.09163531051569076861e-15)},
                {Complex(0.0001285722030382765009958270501823, 2.500441276629004709031733e-10),
                 Complex(7.67848658231026628573152488e-9, 5.00000627469205567279679591433e-6),
                 Complex(3.3142812500162143164933470943e-6, 6.67871532078391980658095553e-9),
                 Complex(2.50050725995913197385188076224e-6, 1.0185769571662125497562811e-10),
                 Complex(2.07092563079747386709534540252e-6, 5.0012995657994976886439659e-10)},
                {Complex(0.0000999868839734954226600621908356, 6.4308798593210535780736734e-9),
                 Complex(-0.00050004307623004565249658650347, -8.333557913579353000279259e-9),
                 Complex(0.000299990502273415816459638775204, -3.168429860092801407152865e-9),
                 Complex(3.744334506143059237157021e-10, 0.0001000601287623185031944481257037),
                 Complex(0.000200025293999464651893192799253, 1.01846943112581124671272e-10)}}},
        TestData{
                 1e-6, 5e-15, // expm1 used specialized algorithm for very small t, therefore we get better error estimates
            Matrix{
                {Complex(7.69290727345543240609440093386e-8, 5.00019048439588280967982e-14),
                 Complex(1.285686760146459388936773128535e-6, 2.39282664517065813864655e-13),
                 Complex(-2.99998302252981975968722077949e-6, 1.00000026941294934013667254718e-6),
                 Complex(-3.749970264471865192901931416e-14, 4.22680832839645429312260505067e-12),
                 Complex(0.00001100001267880340852840823408186, 1.000037950248515255714e-14)},
                {Complex(1.000001088462820721310657157913e-6, -1.102380431374182450429466e-12),
                 Complex(-4.337710455782195101703089e-12, 3.33333341665200889403620316396e-7),
                 Complex(1.000001461116473362305641877344e-6, 3.33333972223542875791524588323e-7),
                 Complex(3.4583324839145745054138e-13, -1.99999902082792470966882056999e-6),
                 Complex(2.00000751000611794025222021112e-6, 3.1666285205570452723468e-13)},
                {Complex(1.000000565844596140862269686628e-7, -1.22122357386673616e-20),
                 Complex(5.00000151180356673349496654633e-8, 8.9583359975070021100051e-15),
                 Complex(3.33332393061225623704451119587e-8, 5.83333570644432275603818e-14),
                 Complex(2.50000007291718614038477134666e-8, -3.99998030648111716555571e-14),
                 Complex(2.00006205839639889010398212124e-8, 6.0944163433981577e-21)},
                {Complex(1.285714362905510938262451383017e-6, 2.5000044064284702727386e-14),
                 Complex(7.69018793298256485467504e-13, 5.00000006250248041049121026427e-8),
                 Complex(3.3331427408035800732932226912e-8, 6.6785728677436375748932e-13),
                 Complex(2.50000507291349566635416468661e-8, 1.00018569730027649227716e-14),
                 Complex(2.00070920174161179028680206981e-8, 5.00001299359228366954767e-14)},
                {Complex(9.99998688460896853318581945651e-7, 6.42859451127988688661156e-13),
                 Complex(-5.00000430714765933921632537022e-6, -8.3333355771599741644861e-13),
                 Complex(2.99999905000227398640188946648e-6, -3.1666842976641256564192e-13),
                 Complex(3.749943350638136716837e-14, 1.00000601250376222809617681308e-6),
                 Complex(2.0000025299940001732460352592e-6, 1.000184623335914060019e-14)}}},
        TestData{
                 1e-8, 5e-15, // expm1 used specialized algorithm for very small t, therefore we get better error estimates
            Matrix{
                {Complex(7.69231368812337259312954179958e-10, 5.000001904837411330787e-18),
                 Complex(1.285714010458789371215269491644e-8, 2.392856837880416791092e-17),
                 Complex(-2.99999983022527495999868931527e-8, 1.00000000269413909713224341929e-8),
                 Complex(-3.749999702643102406286306629e-18, 4.22678594042669919213223726519e-16),
                 Complex(1.100000012678791330788727037782e-7, 1.0000003794985138624e-18)},
                {Complex(1.000000010884615512841495472394e-8, -1.1023809471708892497471e-16),
                 Complex(-4.3376985331291800967994e-16, 3.33333333416666520088549551012e-9),
                 Complex(1.000000014611111647336166029696e-8, 3.33333339722222354287999417899e-9),
                 Complex(3.45833332483912062412e-17, -1.99999999020833279247171859739e-8),
                 Complex(2.000000075100000611793966809e-8, 3.16666628520568665691e-17)},
                {Complex(1.000000005658425118954820243786e-9, -1.2212300928817e-26),
                 Complex(5.00000001511903749953758318104e-10, 8.95833335997601558116e-19),
                 Complex(3.3333332393055612256230143644e-10, 5.833333357064458534841e-18),
                 Complex(2.50000000072916671861400496811e-10, -3.999999803065469540292e-18),
                 Complex(2.00000062058333963988513781463e-10, 6.094444163433e-27)},
                {Complex(1.285714286486172419225828661659e-8, 2.50000004406365108877e-18),
                 Complex(7.6903049405549088988236e-17, 5.0000000006250002480522536035e-10),
                 Complex(3.33333142740086607230006882329e-10, 6.678571442963175658892e-17),
                 Complex(2.5000000507291663495663880804e-10, 1.000001856965775945985e-18),
                 Complex(2.0000070920119598754253535394e-10, 5.000000129935716371214e-18)},
                {Complex(9.99999986884615320454843597534e-9, 6.428571659398353263413e-17),
                 Complex(-5.0000000430714290516489294178e-8, -8.3333333557714022219e-17),
                 Complex(2.99999999050000022739921076087e-8, -3.1666668429765475481e-17),
                 Complex(3.74999943350693881369e-18, 1.00000006012500037622271916761e-8),
                 Complex(2.00000002529999940001803313615e-8, 1.0000018462262621628e-18)}}},
        TestData{
                 1e-12, 5e-15, // expm1 used specialized algorithm for very small t, therefore we get better error estimates
            Matrix{
                {Complex(7.69230769290727387996576312487e-14, 5.00000000019048373e-26),
                 Complex(1.285714285686760164835146459406e-12, 2.39285714282664518e-25),
                 Complex(-2.99999999998302252747252981977e-12, 1.00000000000026941391941294934e-12),
                 Complex(-3.749999999970264308608221865e-26, 4.22678571430832838408120895429e-24),
                 Complex(1.100000000001267879120880340854e-11, 1.00000000003795e-26)},
                {Complex(1.000000000001088461538462820723e-12, -1.102380952380431375e-24),
                 Complex(-4.337698412710455775e-24, 3.33333333333341666666665200885e-13),
                 Complex(1.000000000001461111111116473362e-12, 3.3333333333397222222222354288e-13),
                 Complex(3.4583333333324839e-25, -1.99999999999902083333332792472e-12),
                 Complex(2.00000000000751000000000611794e-12, 3.1666666666285206e-25)},
                {Complex(1.000000000000565842490844596142e-13, -1.22123e-38),
                 Complex(5.00000000000151190476180356681e-14, 8.9583333333359976e-27),
                 Complex(3.33333333332393055555561225623e-14, 5.83333333333570645e-26),
                 Complex(2.50000000000007291666666718614e-14, -3.99999999998030655e-26),
                 Complex(2.00000000006205833333339639885e-14, 6.0944e-39)},
                {Complex(1.28571428571436290293040551094e-12, 2.5000000000044064e-26),
                 Complex(7.690306122330790185e-25, 5.00000000000006250000000248052e-14),
                 Complex(3.33333333314274007936580358014e-14, 6.678571428572867746e-25),
                 Complex(2.50000000000507291666666349566e-14, 1.00000000018569657e-26),
                 Complex(2.00000000070920119047674161139e-14, 5.00000000001299357e-26)},
                {Complex(9.99999999998688461538460896856e-13, 6.42857142859451126e-25),
                 Complex(-5.00000000000430714285714765935e-12, -8.3333333333355771e-25),
                 Complex(2.99999999999905000000000227399e-12, -3.1666666666842977e-25),
                 Complex(3.7499999999433507e-26, 1.000000000006012500000003762227e-12),
                 Complex(2.00000000000252999999999400018e-12, 1.000000000184623e-26)}}}
    };

    for (size_t i = 0; i < testData.size(); ++i)
    {
        TestData& test  = testData[i];
        Matrix arg      = test.t * A;
        Matrix computed = expm1(arg);

        Real t = test.t;

        if (t == 0)
        {
            EXPECT_EQ(computed, test.result) << "i=" << i << " t =" << t;
        }
        else
        {
            EXPECT_LT(maxNorm((computed - test.result) / t), test.errorGoal) << "i=" << i << " t =" << t;
        }
    }
}

TEST(computeZeroEigenvectors, Test)
{
    Matrix A{
        {2, 2,  3,  4},
        {2, 4,  6, 18},
        {3, 6,  9, 12},
        {5, 8, 12, 16}
    };

    Real tol      = 1e-12;
    auto zeroVecA = computeZeroEigenvectors(A, tol);
    EXPECT_EQ(zeroVecA.size(), 1);

    Vector testA = A * zeroVecA[0];
    EXPECT_LT(maxNorm(testA), 1e-14);

    Matrix B{
        {1, 2,  3,  4},
        {2, 4,  6,  8},
        {3, 6,  9, 12},
        {4, 8, 12, 16}
    };

    auto zeroVecB = computeZeroEigenvectors(B, tol);
    EXPECT_EQ(zeroVecB.size(), 3);

    Vector testB = B * zeroVecB[0];
    EXPECT_LT(maxNorm(testB), 1e-14);

    testB = B * zeroVecB[1];
    EXPECT_LT(maxNorm(testB), 1e-14);

    testB = B * zeroVecB[2];
    EXPECT_LT(maxNorm(testB), 1e-14);
}

TEST(superfermionsInResonantLevel, Test)
{
    Real epsilon = 0.5;
    RealVector T{
        {0.3, 2, 0.7}
    };
    RealVector mu{
        {0.1, -0.4, 1.3}
    };
    RealVector Gamma{
        {1, 1.3, 0.1}
    };

    auto model = createModel<ResonantLevel>(epsilon, T, mu, Gamma);

    Matrix Gpm{
        {0, 0,  1, 0}, //
        {0, 0, -1, 0}, //
        {0, 0,  0, 0}, //
        {1, 1,  0, 0}
    };
    Gpm /= std::sqrt(2);

    Matrix Gpp{
        {0, 0, 0, -1}, //
        {0, 0, 0,  1}, //
        {1, 1, 0,  0}, //
        {0, 0, 0,  0}
    };
    Gpp /= std::sqrt(2);

    Matrix Gmp = Gpm.adjoint();
    Matrix Gmm = Gpp.adjoint();

    //
    // Check superfermions
    //
    BlockMatrix Gpm_computed = computeSuperfermion(Keldysh::Plus, Eta::Minus, 0, model);
    BlockMatrix Gpp_computed = computeSuperfermion(Keldysh::Plus, Eta::Plus, 0, model);
    BlockMatrix Gmp_computed = computeSuperfermion(Keldysh::Minus, Eta::Plus, 0, model);
    BlockMatrix Gmm_computed = computeSuperfermion(Keldysh::Minus, Eta::Minus, 0, model);

    EXPECT_EQ(Gpm, Gpm_computed.toDense());
    EXPECT_EQ(Gpp, Gpp_computed.toDense());
    EXPECT_EQ(Gmp, Gmp_computed.toDense());
    EXPECT_EQ(Gmm, Gmm_computed.toDense());

    // Confirm super Pauli principle
    EXPECT_LT(maxNorm(Gpm * Gpm), 1e-16);
    EXPECT_LT(maxNorm(Gpp * Gpp), 1e-16);
    EXPECT_LT(maxNorm(Gmp * Gmp), 1e-16);
    EXPECT_LT(maxNorm(Gmm * Gmm), 1e-16);

    // Zero eigenvector
    const Matrix id = Matrix::Identity(2, 2);
    EXPECT_LT(maxNorm(Gpm * model->vectorize(model->P())), 1e-16);
    EXPECT_LT(maxNorm(Gpp * model->vectorize(model->P())), 1e-16);
    EXPECT_LT(maxNorm(Gmp * model->vectorize(id)), 1e-16);
    EXPECT_LT(maxNorm(Gmm * model->vectorize(id)), 1e-16);

    // Anti commutation
    EXPECT_LT(maxNorm(Gpm * Gmp + Gmp * Gpm - Matrix::Identity(4, 4)), 5e-16);
    EXPECT_LT(maxNorm(Gpm * Gpp + Gpp * Gpm), 1e-16);
    EXPECT_LT(maxNorm(Gpm * Gmm + Gmm * Gpm), 1e-16);

    EXPECT_LT(maxNorm(Gpp * Gmm + Gmm * Gpp - Matrix::Identity(4, 4)), 5e-16);
    EXPECT_LT(maxNorm(Gpp * Gmp + Gmp * Gpp), 1e-16);
    EXPECT_LT(maxNorm(Gpp * Gpm + Gpm * Gpp), 1e-16);

    EXPECT_LT(maxNorm(Gmp * Gpm + Gpm * Gmp - Matrix::Identity(4, 4)), 5e-16);
    EXPECT_LT(maxNorm(Gmp * Gmm + Gmm * Gmp), 1e-16);
    EXPECT_LT(maxNorm(Gmp * Gpp + Gpp * Gmp), 1e-16);

    EXPECT_LT(maxNorm(Gmm * Gpp + Gpp * Gmm - Matrix::Identity(4, 4)), 5e-16);
    EXPECT_LT(maxNorm(Gmm * Gpm + Gpm * Gmm), 1e-16);
    EXPECT_LT(maxNorm(Gmm * Gmp + Gmp * Gmm), 1e-16);

    //
    // Check Liouvillian
    //
    BlockDiagonalMatrix computed = computeLiouvillian(model);
    Matrix analytical            = Complex(0, -1) * epsilon * (Gpp * Gmm - Gpm * Gmp);
    EXPECT_LT(maxNorm(computed.toDense() - analytical), 5e-16);

    //
    // Check SigmaInfty
    //
    auto superfermions             = computeAllSuperfermions(Keldysh::Plus, model);
    auto superfermionsAnnihilation = computeAllSuperfermions(Keldysh::Minus, model);

    computed      = computeSigmaInfty(superfermions, superfermionsAnnihilation, model);
    Real GammaSum = Gamma.sum();
    analytical    = -GammaSum / 2 * (Gpp * Gmm + Gpm * Gmp);
    EXPECT_LE(maxNorm(computed.toDense() - analytical), 5e-16);

    //
    // Check SigmaInftyCurrent
    //
    RowVector computed0 = computeSigmaInftyCurrent(0, superfermionsAnnihilation, model);
    RowVector computed1 = computeSigmaInftyCurrent(1, superfermionsAnnihilation, model);
    RowVector computed2 = computeSigmaInftyCurrent(2, superfermionsAnnihilation, model);

    Vector idCol          = model->vectorize(id);
    RowVector idRow       = idCol.transpose();
    RowVector analytical0 = -1.0 / 4.0 * Gamma[0] * idRow * (Gmp * Gmm - Gmm * Gmp);
    RowVector analytical1 = -1.0 / 4.0 * Gamma[1] * idRow * (Gmp * Gmm - Gmm * Gmp);
    RowVector analytical2 = -1.0 / 4.0 * Gamma[2] * idRow * (Gmp * Gmm - Gmm * Gmp);

    EXPECT_LE(maxNorm(computed0 - analytical0), 5e-16)
        << "\ncomputed = " << computed0 << "\nanalytical = " << analytical0;
    EXPECT_LE(maxNorm(computed1 - analytical1), 5e-16)
        << "\ncomputed = " << computed1 << "\nanalytical = " << analytical1;
    EXPECT_LE(maxNorm(computed2 - analytical2), 5e-16)
        << "\ncomputed = " << computed2 << "\nanalytical = " << analytical2;

    //
    // Check computeGammaGG
    //
    // Test continuity around t=0, will only catch rough errors
    BlockDiagonalMatrix ggg0     = computeGammaGG(0, superfermions, model);
    BlockDiagonalMatrix gggSmall = computeGammaGG(1e-6, superfermions, model);
    EXPECT_LT(maxNorm(ggg0.toDense() - gggSmall.toDense()), 1e-11);

    // Test at t=0
    Matrix expected0 = -2.0 / M_PI * (Gamma[0] * mu[0] + Gamma[1] * mu[1] + Gamma[2] * mu[2]) * Gpp * Gpm;
    EXPECT_LT(maxNorm(ggg0.toDense() - expected0), 1e-16);

    // Test at some finite t
    Real t                          = 1.14;
    BlockDiagonalMatrix computedGG1 = computeGammaGG(t, superfermions, model);
    Matrix expectedGG1              = -2.0 *
                         (Gamma[0] * T[0] * sin(mu[0] * t) / sinh(M_PI * t * T[0]) +
                          Gamma[1] * T[1] * sin(mu[1] * t) / sinh(M_PI * t * T[1]) +
                          Gamma[2] * T[2] * sin(mu[2] * t) / sinh(M_PI * t * T[2])) *
                         Gpp * Gpm;
    EXPECT_LT(maxNorm(computedGG1.toDense() - expectedGG1), 1e-16);
}